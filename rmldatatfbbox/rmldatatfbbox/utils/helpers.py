import os
import json
import cv2
from pathlib import Path
import tensorflow as tf
from object_detection.utils import dataset_util

### FUNCTIONS ###
def construct_all(image_ids: list, temp_dir: Path):
    labeled_images = {}
    label_to_int_dict = {}

    for image_id in image_ids:
        labeled_images[image_id] = construct(image_id, temp_dir, label_to_int_dict)
    
    return labeled_images, label_to_int_dict

def construct(image_id: str, temp_dir: Path, label_to_int_dict: dict):
    if temp_dir is None:
        cwd = Path.cwd()
        data_dir = cwd / 'data'
    else:
        data_dir = temp_dir

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
        if label not in label_to_int_dict.keys():
            label_to_int_dict[label] = len(label_to_int_dict)
    
    bboxes = {"image_id": image_id, "image_filepath": image_filepath, "image_type": image_type, "label_boxes": label_boxes, "xdim": xdim, "ydim": ydim}

    return bboxes

def write_label_map(dataset_name, out_dir, label_to_int_dict):
        """Writes out the TensorFlow Object Detection Label Map
        
        Args:
            dataset_name (str): the name of the dataset
        """
        dataset_path = out_dir / 'dataset' / dataset_name
        label_map_filepath = dataset_path / 'label_map.pbtxt'
        label_map = []
        for label_name, label_int in label_to_int_dict.items():
            label_info = "\n".join([
                "item {", "  id: {id}".format(id=label_int),
                "  name: '{name}'".format(name=label_name), "}"
            ])
            label_map.append(label_info)
        with open(label_map_filepath, 'w') as outfile:
            outfile.write("\n\n".join(label_map))

def export_as_TFExample(object, label_to_int_dict):
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
            classes.append(label_to_int_dict[bounding_box["label"]])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image_height),
            'image/width': dataset_util.int64_feature(image_width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tf_example
