import os
import json
import cv2
from pathlib import Path
import tensorflow as tf
from ravenml.data.write_dataset import DatasetWriter
from ravenml.data.interfaces import CreateInput
from ravenml.utils.question import cli_spinner_wrapper


class BboxDatasetWriter(DatasetWriter):

    def __init__(self, create: CreateInput, **kwargs):
        # Additional initializations of plugin specific variables can go here
        super().__init__(create, **kwargs)

    def construct_all(self):
        labeled_images = {}
        
        for image_id in self.image_ids:
            labeled_images[image_id] = self.construct(image_id)
        
        return labeled_images

    def construct(self, image_id: str):
        data_dir = self.temp_dir

        image_filepath = None
        image_type = None
        file_extensions = [".png", ".jpg", ".jpeg"]
        for extension in file_extensions:
            if os.path.exists(data_dir / f'image_{image_id}{extension}'):
                image_filepath = data_dir / f'image_{image_id}{extension}'
                image_type = extension
                ydim, xdim = tuple(cv2.imread(str(image_filepath.absolute())).shape[:2])
                break

        if image_filepath is None:
            raise ValueError("Hmm, there doesn't seem to be a valid image filepath.")

        with open(data_dir / f"meta_{image_id}.json", "r") as f:
            meta = json.load(f)

        label_boxes = []
        for label, b in meta['bboxes'].items():
            label_boxes.append({"label": label, "xmin": b['xmin'], "xmax": b['xmax'], "ymin": b['ymin'], "ymax": b['ymax']})
            if label not in self.label_to_int_dict.keys():
                self.label_to_int_dict[label] = len(self.label_to_int_dict)
        
        bboxes = {"image_id": image_id, "image_filepath": image_filepath, "image_type": image_type, "label_boxes": label_boxes, "xdim": xdim, "ydim": ydim}

        return bboxes

    def export_as_TFExample(self, object):
            """Converts LabeledImageMask object to tf_example
            
            Returns:
                tf_example (tf.train.Example): TensorFlow specified training object.
            """
            path_to_image = Path(object["image_filepath"])

            with tf.io.gfile.GFile(str(path_to_image), 'rb') as fid:
                encoded_png = fid.read()

            image_width  = object["xdim"]
            image_height = object["ydim"]

            filename = path_to_image.name.encode('utf8')
            image_format = bytes(object["image_type"], encoding='utf-8')
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []

            for bounding_box in object["label_boxes"]:
                xmins.append(bounding_box["xmin"] / image_width)
                xmaxs.append(bounding_box["xmax"] / image_width)
                ymins.append(bounding_box["ymin"] / image_height)
                ymaxs.append(bounding_box["ymax"] / image_height)
                classes_text.append(bounding_box["label"].encode('utf8'))
                classes.append(self.label_to_int_dict[bounding_box["label"]])

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
                'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
                'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
                'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_png])),
                'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
                'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
                'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
            }))

            return tf_example

    @cli_spinner_wrapper("Writing additional files...")
    def write_additional_files(self):
            """Writes out the TensorFlow Object Detection Label Map
            
            Args:
                dataset_name (str): the name of the dataset
            """
            dataset_path = self.dataset_path / self.dataset_name
            label_map_filepath = dataset_path / 'label_map.pbtxt'
            label_map = []
            for label_name, label_int in self.label_to_int_dict.items():
                label_info = "\n".join([
                    "item {", "  id: {id}".format(id=label_int),
                    "  name: '{name}'".format(name=label_name), "}"
                ])
                label_map.append(label_info)
            with open(label_map_filepath, 'w') as outfile:
                outfile.write("\n\n".join(label_map))
