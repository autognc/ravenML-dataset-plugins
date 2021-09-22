import os
import numpy as np
from pathlib import Path
from ravenml.data.write_dataset import DefaultDatasetWriter
from ravenml.data.interfaces import CreateInput
from ravenml.utils.question import user_selects
from ravenml.data.helpers import split_data
import shutil
import glob

class PTGANDatasetWriter(DefaultDatasetWriter):
    """Inherits from DefaultDatasetWriter, handles
        dataset creation
    """

    def __init__(self, create: CreateInput):
        """Initialization inherited from DatasetWriter,
            passes associated_files

        Variables (not in DefaultDatasetWriter):
            label_to_int_dict (dict): dict with labels and corresponding
                unique ints
        """
        super().__init__(create)
        self.A_paths, self.B_paths = create.A_paths, create.B_paths

    def write_dataset(self, extensions):

        dataset_path = self.dataset_path / self.dataset_name

        A_data = self.grab_paths(self.A_paths, extensions)
        A_train, A_test =  split_data(A_data, self.test_percent)
        
        train_path = dataset_path / 'trainA'
        test_path = dataset_path / 'testA'
        self.write_out_split(train_path, A_train)
        self.write_out_split(test_path, A_test)


        B_data = self.grab_paths(self.B_paths, extensions)
        B_train, B_test =  split_data(B_data, self.test_percent)

        train_path = dataset_path / 'trainB'
        test_path = dataset_path / 'testB'
        self.write_out_split(train_path, B_train)
        self.write_out_split(test_path, B_test)


    
    def write_out_split(self, path, objects):
        os.mkdir(path)
        for obj in objects:
            shutil.copy(obj, path)
    
    def grab_paths(self, imageset_paths, extensions):
        paths = []
        for imageset_path in imageset_paths:
            for ext in extensions:
                paths.extend(
                    list(glob.glob(
                        str(imageset_path / f"[!mask]*{ext}")
                    ))
                )
        return paths

def partition_imagesets(imagesets, A, B):
    
    unassigned_imagesets = [
        i for i in imagesets
        if i not in A and i not in B
    ]
    if len(unassigned_imagesets) == 0:
        return A,B
    choices = user_selects(
        "Found unassigned_imagesets. Choose imagesets to place in A:", 
        unassigned_imagesets,
        selection_type="checkbox"
    )
    A.extend(choices)
    B.extend([i for i in unassigned_imagesets if i not in choices])

    return A, B
