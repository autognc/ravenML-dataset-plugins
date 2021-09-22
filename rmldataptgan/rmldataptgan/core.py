"""
Author(s):      Carson Schubert (carson.schubert14@gmail.com)
Date Created:   04/22/2020

Core command group and commands for PT GAN dataset plugin.
"""
import click 
from rmldataptgan.utils.helpers import PTGANDatasetWriter, partition_imagesets
from ravenml.data.options import pass_create
from ravenml.data.interfaces import CreateInput, CreateOutput
from ravenml.utils.question import user_selects
import os
from itertools import chain
### COMMANDS ###
@click.command(help='Create a dataset in ptgan format.')
@pass_create
@click.pass_context
def pt_gan(ctx, create: CreateInput):
    """Main driver of file, creates tf_record dataset

    Args:
        ctx (Context): click context object
        create (CreateInput): input for dataset creation
    """
    config = create.config["plugin"]
    imagesets_A =  config.get("imagesets_A", [])
    imagesets_B = config.get("imagesets_B", [])
    all_imagesets = {os.path.basename(i):i for i in create.imageset_paths}
    for imgset in chain(imagesets_A, imagesets_B):
        if imgset not in all_imagesets.keys() and\
            imgset not in all_imagesets.values():
            raise ValueError(f"{imgset} listed in A or B but not in imageset field of config file")
    imagesets_A, imagesets_B = partition_imagesets(list(map(str, all_imagesets.keys())), imagesets_A, imagesets_B)

    create.A_paths = [all_imagesets[name] for name in imagesets_A]
    create.B_paths = [all_imagesets[name] for name in imagesets_B] 

    extensions = [
        '.jpg',
        '.png'
    ]
    
    datasetWriter = PTGANDatasetWriter(create)

    datasetWriter.write_dataset(extensions)
    datasetWriter.write_metadata()


    return CreateOutput()
