import os
from setuptools import setup, find_packages

# figured out how to add object-detection via:
# https://stackoverflow.com/questions/12518499/pip-ignores-dependency-links-in-setup-py

# figured out to use find_packages() via:
# https://stackoverflow.com/questions/10924885/is-it-possible-to-include-subdirectories-using-dist-utils-setup-py-as-part-of

# determine GPU or CPU install via env variable
# gpu = os.getenv('RML_BBOX_GPU')
# tensorflow_pkg = 'tensorflow==1.14.0' if not gpu else 'tensorflow-gpu==1.14.0'

setup(
    name='rmldatatfbbox',
    version='0.1',
    description='Dataset creation plugin for ravenML',
    packages=find_packages(),
    install_requires=[
        'scikit-learn==0.20.2',
        'object-detection @ https://github.com/autognc/object-detection/tarball/object-detection#egg=object-detection',
        'opencv-python==4.1.2.30',
        'requests',
        'pandas<1.0'
    ],
    entry_points='''
        [ravenml.plugins.data]
        tf_bbox=rmldatatfbbox.core:tf_bbox
    '''
)
