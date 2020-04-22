"""
Author(s):      Carson Schubert (carson.schubert14@gmail.com)
Date Created:   04/22/2020

Core command group and commands for TF Bounding Box dataset plugin.
"""
import click 

### OPTIONS ###


### COMMANDS ###
@click.command(help='TensorFlow Object Detection with bounding boxes.')
@click.pass_context
def tf_bbox(ctx):
    pass
    
# @tf_bbox.command(help='Create a dataset.')
# def create(ctx):
#     click.echo("Dataset gets created here.")

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
