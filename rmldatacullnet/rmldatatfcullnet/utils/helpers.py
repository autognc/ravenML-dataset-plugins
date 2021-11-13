import os
import json
import numpy as np
import tqdm
from ravenml.data.write_dataset import DefaultDatasetWriter
from ravenml.data.interfaces import CreateInput
from ravenml.data.helpers import copy_associated_files
from pathlib import Path
import shutil

default_associated_files = [ 
        ('meta_', '.json'),
        ('image_', '.png'),
        ('image_', '.jpg'),
        ('image_', '.jpeg'),
    ]
class CullNetDatasetWriter(DefaultDatasetWriter):
    """Inherits from DefaultDatasetWriter, handles
        dataset creation

    Methods (not in DefaultDatasetWriter):
        construct (image_id): helper method for construct_all
        export_data (object): helper method for write_data
    """

    def __init__(self, create: CreateInput, associated_files=None):
        """Initialization inherited from DatasetWriter,
            passes associated_files

        Variables (not in DefaultDatasetWriter):
            label_to_int_dict (dict): dict with labels and corresponding
                unique ints
        """
        super().__init__(create)
        if associated_files is None:
            self.associated_files = default_associated_files
        self.keypoints = None
        for imageset_path in self.imageset_paths:
            with open(os.path.join(imageset_path, 'metadata.json'), 'r') as f:
                keypoints = np.array(json.load(f)['keypoints'])
            if self.keypoints is None:
                self.keypoints = keypoints
            else:
                if not np.all(np.abs(self.keypoints - keypoints) < 1e-5):
                    raise ValueError('Imagesets have non-matching 3D keypoints')

    def construct_all(self):
        """Constructs objects for all data passed to it, sets obj_dict
            to dict of constructed objects

        Variables needed:
            image_ids (list): list of image_ids (tuples) to create objects for
        """
        labeled_images = {}

        for image_id in tqdm.tqdm(self.image_ids, "Constructing data objects"):
            labeled_images[image_id] = self.construct(image_id)
        
        self.obj_dict = labeled_images

    def construct(self, image_id: tuple):
        """Helper function for construct_all, creates individual object
            for passed image_id

        Args:
            image_id (tuple): image_id to create object for, tuple of path and
                image_id
        Variables needed:
            label_to_int_dict (dict): dict with labels and corresponding
                unique ints
        Returns:
            bboxes (object): object corresponding to the given image_id
        """
        data_dir = image_id[0]

        image_filepath = None
        image_type = None
        file_extensions = [".png", ".jpg", ".jpeg"]
        for extension in file_extensions:
            if os.path.exists(data_dir / f'image_{image_id[1]}{extension}'):
                image_filepath = data_dir / f'image_{image_id[1]}{extension}'
                image_type = extension
                break

        if image_filepath is None:
            raise ValueError("Hmm, there doesn't seem to be a valid image filepath.")
        return image_id

    def write_out_train_split(self, objects, path, split_type='train'):
        """Writes out list of objects out as a single tf_example
        
        Args:
            objects (list): list of objects to put into the tf_example 
            path (Path): directory to write this tf_example to, encompassing the name
            split_type (str, optional): specific split_type to be written to
        """
        copy_associated_files(objects, path, self.associated_files)
    def write_additional_files(self):
        """Writes out the TensorFlow Object Detection Label Map
        
        Variables needed:
            dataset_path (Path): path to dataset
            dataset_name (str): the name of the dataset
            label_to_int_dict (dict): dict with labels and corresponding
                unique ints
        """
        dataset_path = self.dataset_path / self.dataset_name
        np.save(str(dataset_path / 'keypoints.npy'), self.keypoints)
    def write_extra_files(self, extra_files):
        """Copies extra files to dataset
        """
        dataset_path = self.dataset_path / self.dataset_name
        for extra_file in extra_files:
            if not os.path.isfile(extra_file):
                raise ValueError(f'{extra_file} is not a file')
            shutil.copy(extra_file, dataset_path.absolute())

