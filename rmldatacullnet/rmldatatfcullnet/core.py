"""
Author(s):      Jacob Deutsch (jacobdeutsch@utexas.edu)
Date Created:   11/11/2021

Core command group and commands for cullnet dataset plugin
"""
import click 
from rmldatatfcullnet.utils.helpers import CullNetDatasetWriter
from ravenml.data.options import pass_create
from ravenml.data.interfaces import CreateInput, CreateOutput


### COMMANDS ###
@click.command(help='Create a dataset in cullnet format.')
@pass_create
@click.pass_context
def tf_cullnet(ctx, create: CreateInput):
    """Main driver of file, creates cullnet dataset

    Args:
        ctx (Context): click context object
        create (CreateInput): input for dataset creation
    """
    config = create.config["plugin"]
    associated_files = [ 
        ('meta_', '.json'),
        ('image_', '.png'),
        ('image_', '.jpg'),
        ('image_', '.jpeg'),
    ]

    metadata_format = ('meta_', '.json')  
    datasetWriter = CullNetDatasetWriter(create)
    datasetWriter.write_extra_files(config.get("extra_files", []))
    datasetWriter.load_image_ids(metadata_format)
    
    # Filtering
    if config.get('setSizeFilter'):
        datasetWriter.set_size_filter(config['setSizeFilter'])
    if config.get('tagFilter'):
        datasetWriter.interactive_tag_filter()
            
    datasetWriter.construct_all()

    datasetWriter.write_dataset(associated_files)
    datasetWriter.write_metadata()
    datasetWriter.write_additional_files()
    
    return CreateOutput()
