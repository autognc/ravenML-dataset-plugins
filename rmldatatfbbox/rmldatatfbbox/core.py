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
    config = create.config["plugin"]

    set up TF verbosity
    if config.get('verbose'):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    else:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    associated_files = {
        "image_type_1": ".png",
        "image_type_2": ".jpg",
        "image_type_3": ".jpeg",
        "metadata": ".json",
        "labels": ".csv",
        "PASCAL_VOC_labels": ".xml"
    }

    related_data_prefixes = {
        'images': 'image_',
        'labels': 'bboxLabels_'
    }

    datasetWriter = BboxDatasetWriter(create, associated_files=associated_files, related_data_prefixes=related_data_prefixes)

    datasetWriter.filter_sets('meta_')
    datasetWriter.load_data(metadata_prefix='meta_')
            
    labeled_images = datasetWriter.construct_all()

    datasetWriter.write_dataset(list(labeled_images.values()))
    datasetWriter.write_metadata()
    datasetWriter.write_additional_files()

    return CreateOutput()
