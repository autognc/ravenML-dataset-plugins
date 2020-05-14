"""
Author(s):      Carson Schubert (carson.schubert14@gmail.com)
Date Created:   04/22/2020

Core command group and commands for TF Bounding Box dataset plugin.
"""
import click 
import os
import shutil
from pathlib import Path
from ravenml.options import verbose_opt
from ravenml.utils.question import user_input, cli_spinner, user_confirms
# from ravenml.utils.plugins import fill_basic_dataset_creation_metadata
from rmldatatfbbox.utils.helpers import (default_filter_and_load, construct_all,
                                         write_label_map)
from rmldatatfbbox.utils.write_dataset import write_dataset, write_metadata
from rmldatatfbbox.utils.io_utils import upload_dataset

### OPTIONS ###
### OPTIONS: could be used for other plugins###
name_opt = click.option(
    '-n', '--name', type=str, 
    help='First and Last name of user.'
)

comments_opt = click.option(
    '--comments', type=str, 
    help='Comments about the training.'    
)

dataset_name_opt = click.option(
    '--dataset-name', type=str,
    help='Name of dataset being created. Ignored without --no-user.'
)

kfolds_opt = click.option(
    '--kfolds', '-k', type=int, default=-1, is_eager=True,
    help='Number of folds in dataset. Ignored without --no-user.'
)

upload_opt = click.option(
    '--upload', type=str, is_eager=True,
    help='Enter the bucket you would like to upload too. Ignored without --no-user.'
)

delete_local_opt = click.option(
    '--delete-local', '-d', 'delete_local', is_flag=True, is_eager=True,
    help='Enter your first and last name. Ignored without --no-user.'
)

filter_opt = click.option(
    '-f', '--filter', is_flag=True, is_eager=True,
    help='Enable interactive image filtering option'
)

### COMMANDS ###
# @click.command(help='TensorFlow Object Detection with bounding boxes.')
# @click.pass_context
@click.group()
def tf_bbox():#(ctx):
    pass

#temporary, needs to be changed to config input    
@tf_bbox.command(help='Create a dataset.')
@verbose_opt
@name_opt
@comments_opt
@dataset_name_opt
@kfolds_opt
@upload_opt
@delete_local_opt
@filter_opt
# Will have additional params, ctx and something like create_dataset: CreateDatasetInput,
def create(verbose: bool, 
            name:str, 
            comments:str, 
            dataset_name: str, 
            kfolds: int, 
            upload: str,
            delete_local: bool, 
            filter: bool):

    # Hardcoded Values that are assumed to be inputs
    data_origin="S3"
    bucket="skr-images-training"
    user_folder_selection=("jigsaw_tester",)
    data_path=None

    # create dataset creation metadata dict and populate with basic information
    # Probably needs to be a new function in ravenml for dataset creation metadata
    # Commented out dataset type
    metadata = {}
    # fill_basic_dataset_creation_metadata(metadata, user_folder_selection, name, comments)

    if data_origin == "Local":
        image_ids, filter_metadata = default_filter_and_load(
                data_source=data_origin, data_filepath=data_path, filter=filter)
    else:
        image_ids, filter_metadata, temp_dir = default_filter_and_load(
            data_source=data_origin, bucket=bucket, filter_vals=user_folder_selection, filter=filter)
    
    # Transformation is Missing
    transform_metadata = []

    dataset_name = dataset_name if dataset_name else user_input(message="What would you like to name this dataset?")
                                                # ,validator=FilenameValidator)
    k_folds_specified = kfolds if kfolds != -1 else user_input(message="How many folds would you like the dataset to have?",
                                                            #    validator=IntegerValidator,
                                                               default="5")
    
    labeled_images = construct_all(image_ids, temp_dir=temp_dir)
    cli_spinner("Writing out dataset locally...", write_dataset,list(labeled_images.values()),
                                                                     custom_dataset_name=dataset_name,
                                                                     num_folds=int(k_folds_specified))
    cli_spinner("Writing out metadata locally...", write_metadata,name=dataset_name,
                                                                  user=name,
                                                                  comments=comments,
                                                                  training_type="Bounding_Box",
                                                                  image_ids=image_ids,
                                                                  filters=filter_metadata,
                                                                  transforms=transform_metadata)
    cli_spinner("Writing out additional files...", write_label_map,dataset_name)
    cli_spinner("Deleting temp directory...", shutil.rmtree, temp_dir)

    if (upload or user_confirms(message="Would you like to upload the dataset to S3?")):
        default = ""
        default = os.getenv('DATASETS_BUCKET_NAME', default)
        bucket = upload if upload else user_input(message="Which bucket would you like to upload to?", default=default)
        cli_spinner("Uploading dataset to S3...", upload_dataset, bucket_name=bucket, directory=Path.cwd() / 'dataset' / dataset_name)

    if (dataset_name != '') and (delete_local or user_confirms(
            message="Would you like to delete your " + dataset_name + " dataset?")):
        dataset_path = Path.cwd() / 'dataset' / dataset_name

        cli_spinner("Deleting " + dataset_name + " dataset...", shutil.rmtree, dataset_path)

tf_bbox()

### HELPERS ###

# # stdout redirection found at https://codingdose.info/2018/03/22/supress-print-output-in-python/
# def _import_od():
#     """ Imports the necessary libraries for object detection training.
#     Used to avoid importing them at the top of the file where they get imported
#     on every ravenML command call, even those not to this plugin.
    
#     Also suppresses warning outputs from the TF OD API.
#     """
#     # create a text trap and redirect stdout
#     # to suppress printed warnings from object detection and tf
#     text_trap = io.StringIO()
#     sys.stdout = text_trap
#     sys.stderr = text_trap
    
#     # Calls to _dynamic_import below map to the following standard imports:
#     #
#     # import tensorflow as tf
#     # from object_detection import model_hparams
#     # from object_detection import model_lib
#     _dynamic_import('tensorflow', 'tf')
#     _dynamic_import('object_detection.model_hparams', 'model_hparams')
#     _dynamic_import('object_detection.model_lib', 'model_lib')
#     _dynamic_import('object_detection.exporter', 'exporter')
#     _dynamic_import('object_detection.protos', 'pipeline_pb2', asfunction=True)
    
#     # now restore stdout function
#     sys.stdout = sys.__stdout__
#     sys.stderr = sys.__stderr__
    
# # this function is derived from https://stackoverflow.com/a/46878490
# # NOTE: this function should be used in all plugins, but the function is NOT
# # importable because of the use of globals(). You must copy the code.
# def _dynamic_import(modulename, shortname = None, asfunction = False):
#     """ Function to dynamically import python modules into the global scope.

#     Args:
#         modulename (str): name of the module to import (ex: os, ex: os.path)
#         shortname (str, optional): desired shortname binding of the module (ex: import tensorflow as tf)
#         asfunction (bool, optional): whether the shortname is a module function or not (ex: from time import time)
        
#     Examples:
#         Whole module import: i.e, replace "import tensorflow"
#         >>> _dynamic_import('tensorflow')
        
#         Named module import: i.e, replace "import tensorflow as tf"
#         >>> _dynamic_import('tensorflow', 'tf')
        
#         Submodule import: i.e, replace "from object_detction import model_lib"
#         >>> _dynamic_import('object_detection.model_lib', 'model_lib')
        
#         Function import: i.e, replace "from ravenml.utils.config import get_config"
#         >>> _dynamic_import('ravenml.utils.config', 'get_config', asfunction=True)
        
#     """
#     if shortname is None: 
#         shortname = modulename
#     if asfunction is False:
#         globals()[shortname] = importlib.import_module(modulename)
#     else:        
#         globals()[shortname] = getattr(importlib.import_module(modulename), shortname)
