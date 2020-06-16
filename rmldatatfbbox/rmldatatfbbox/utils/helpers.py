import os
import json
import cv2
from pathlib import Path
from rmldatatfbbox.utils.classes import BoundingBox, BBoxLabeledImage

### FUNCTIONS ###
def construct_all(image_ids, **kwargs):
    labeled_images = {}

    for image_id in image_ids:
        labeled_images[image_id] = construct(image_id, **kwargs)
    
    return labeled_images

def construct(image_id, **kwargs):
    temp_dir = kwargs['temp_dir']
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
        # missing param for bboxLabeledImage class
        label_boxes.append(BoundingBox(label, b['xmin'], b['xmax'], b['ymin'], b['ymax']))
        add_label_int(label)
    
    bbox = BBoxLabeledImage(image_id, image_filepath, image_type, label_boxes, xdim, ydim)

    return bbox

def add_label_int(label_to_add):
    if label_to_add not in BBoxLabeledImage._label_to_int_dict.keys():
        # add the new label
        BBoxLabeledImage._label_to_int_dict[label_to_add] = None

        # renumber all values
        BBoxLabeledImage.renumber_label_to_int_dict()

def write_label_map(dataset_name):
        """Writes out the TensorFlow Object Detection Label Map
        
        Args:
            dataset_name (str): the name of the dataset
        """
        dataset_path = Path.cwd() / 'dataset' / dataset_name
        label_map_filepath = dataset_path / 'label_map.pbtxt'
        label_map = []
        for label_name, label_int in BBoxLabeledImage._label_to_int_dict.items():
            label_info = "\n".join([
                "item {", "  id: {id}".format(id=label_int),
                "  name: '{name}'".format(name=label_name), "}"
            ])
            label_map.append(label_info)
        with open(label_map_filepath, 'w') as outfile:
            outfile.write("\n\n".join(label_map))
