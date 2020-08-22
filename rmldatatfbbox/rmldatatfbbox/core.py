"""
Author(s):      Carson Schubert (carson.schubert14@gmail.com)
Date Created:   04/22/2020

Core command group and commands for TF Bounding Box dataset plugin.
"""
import click 
import sys
import io
import importlib
import shutil
import tensorflow as tf
from pathlib import Path
from rmldatatfbbox.utils.helpers import BboxDatasetWriter
from ravenml.utils.question import user_input, cli_spinner
from ravenml.data.options import pass_create
from ravenml.data.interfaces import CreateInput, CreateOutput

### COMMANDS ###
@click.command(help='Create a dataset for TensorFlow Object Detection with bounding boxes.')
@pass_create
@click.pass_context
def tf_bbox(ctx, create: CreateInput):
    """Main driver of file, creates tf_bbox dataset

    Args:
        ctx (Context): click context object
        create (CreateInput): input for dataset creation
    """
    config = create.config["plugin"]

    # set up TF verbosity
    if config.get('verbose'):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    else:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    associated_files = [ 
        ('meta_', '.json'),
        ('image_', '.png'),
        ('image_', '.jpg'),
        ('image_', '.jpeg'),
        ('bboxLabels_', '.csv')
    ]

    metadata_format = ('meta_', '.json')
    
    datasetWriter = BboxDatasetWriter(create)

    datasetWriter.load_image_ids(metadata_format)

    if config.get('filter'):
        datasetWriter.interactive_filter()
            
    datasetWriter.construct_all()

    datasetWriter.write_dataset(associated_files)
    datasetWriter.write_metadata()
    datasetWriter.write_additional_files()

    return CreateOutput()
