# import os, shutil
# import tensorflow as tf
# from pathlib import Path
# from object_detection.utils import dataset_util

# ### CLASSES ### LabeledImage should probably be global
# class LabeledImage():
#     def __init__(self, image_id):
#         self.image_id = image_id
    
#     @property
#     @classmethod
#     def temp_dir(cls):
#         raise NotImplementedError

#     @property
#     @classmethod
#     def associated_files(cls):
#         raise NotImplementedError

#     @property
#     @classmethod
#     def related_data_prefixes(cls):
#         raise NotImplementedError

#     def copy_associated_files(self, destination, **kwargs):
#         if self.temp_dir is None:
#             data_dir = Path.cwd() / "data"
#         else:
#             data_dir = self.temp_dir
#         for suffix in self.associated_files.values():
#             for prefix in self.related_data_prefixes.values():
#                 filepath = data_dir / f'{prefix}{self.image_id}{suffix}'
#                 if os.path.exists(filepath):
#                     shutil.copy(
#                         str(filepath.absolute()), str(destination.absolute()))

# class BoundingBox:
#     """Stores the label and bounding box dimensions for a detected image region
    
#     Attributes:
#         label (str): the classification label for the region (e.g., "cygnus")
#         xmin (int): the pixel location of the left edge of the bounding box
#         xmax (int): the pixel location of the right edge of the bounding box
#         ymin (int): the pixel location of the top edge of the bounding box
#         ymax (int): the pixel location of the top edge of the bounding box
#     """

#     def __init__(self, label, xmin, xmax, ymin, ymax):
#         self.label = label
#         self.xmin = xmin
#         self.xmax = xmax
#         self.ymin = ymin
#         self.ymax = ymax
#         # self.label_int_class = label_int_class Probably causing an issue

#     def __repr__(self):
#         return "label: {} | xmin: {} | xmax: {} | ymin: {} | ymax: {}".format(
#             self.label, self.xmin, self.xmax, self.ymin, self.ymax)
    
#     # @property
#     # def label_int(self):
#     #     return self.label_int_class._label_to_int_dict[self.label]

# class BBoxLabeledImage(LabeledImage):
#     """Stores bounding-box-labeled image data and provides related operations

#     Attributes:
#         image_id (str): the unique ID for the image and labeled data
#         image_path (str): the path to the source image
#         label_boxes (list): a list of BoundingBox objects that store the labels
#             and dimensions of each bounding box in the image
#         xdim (int): width of the image (in pixels)
#         ydim (int): height of the image (in pixels)
#     """
#     _label_to_int_dict = {}

#     associated_files = {
#         "image_type_1": ".png",
#         "image_type_2": ".jpg",
#         "image_type_3": ".jpeg",
#         "metadata": ".json",
#         "labels": ".csv",
#         "PASCAL_VOC_labels": ".xml"
#     }
    
#     related_data_prefixes = {
#         'images': 'image_',
#         'labels': 'bboxLabels_'
#     }

#     temp_dir = None

#     training_type = "Bounding Box"
    
#     def __init__(self, image_id, image_path, image_type, label_boxes, xdim, ydim):
#         super().__init__(image_id)
#         self.image_id = image_id
#         self.image_path = image_path
#         self.image_type = image_type
#         self.label_boxes = label_boxes
#         self.xdim = xdim
#         self.ydim = ydim

#     ## HELPERS ##
#     @classmethod
#     def renumber_label_to_int_dict(cls):
#         for i, label in enumerate(BBoxLabeledImage._label_to_int_dict.keys()):
#             BBoxLabeledImage._label_to_int_dict[label] = i + 1
    
#     def export_as_TFExample(self):
#         """Converts LabeledImageMask object to tf_example
        
#         Returns:
#             tf_example (tf.train.Example): TensorFlow specified training object.
#         """
#         path_to_image = Path(self.image_path)

#         with tf.io.gfile.GFile(str(path_to_image), 'rb') as fid:
#             encoded_png = fid.read()

#         image_width  = self.xdim
#         image_height = self.ydim

#         filename = path_to_image.name.encode('utf8')
#         image_format = bytes(self.image_type, encoding='utf-8')
#         xmins = []
#         xmaxs = []
#         ymins = []
#         ymaxs = []
#         classes_text = []
#         classes = []

#         for bounding_box in self.label_boxes:
#             xmins.append(bounding_box.xmin / image_width)
#             xmaxs.append(bounding_box.xmax / image_width)
#             ymins.append(bounding_box.ymin / image_height)
#             ymaxs.append(bounding_box.ymax / image_height)
#             classes_text.append(bounding_box.label.encode('utf8'))
#             # classes.append(bounding_box.label_int) # Probably causing an issue

#         tf_example = tf.train.Example(features=tf.train.Features(feature={
#             'image/height': dataset_util.int64_feature(image_height),
#             'image/width': dataset_util.int64_feature(image_width),
#             'image/filename': dataset_util.bytes_feature(filename),
#             'image/source_id': dataset_util.bytes_feature(filename),
#             'image/encoded': dataset_util.bytes_feature(encoded_png),
#             'image/format': dataset_util.bytes_feature(image_format),
#             'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#             'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#             'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#             'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#             'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#             'image/object/class/label': dataset_util.int64_list_feature(classes),
#         }))

#         return tf_example
