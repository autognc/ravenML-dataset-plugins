from setuptools import setup, find_packages

# figured out to use find_packages() via:
# https://stackoverflow.com/questions/10924885/is-it-possible-to-include-subdirectories-using-dist-utils-setup-py-as-part-of

# determine GPU or CPU install via env variable
# gpu = os.getenv('RML_BBOX_GPU')
# tensorflow_pkg = 'tensorflow==1.14.0' if not gpu else 'tensorflow-gpu==1.14.0'

setup(
    name='rmldatatfrecord',
    version='0.1',
    description='Dataset creation plugin for ravenML',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'tensorflow',
    ],
    entry_points='''
        [ravenml.plugins.data]
        tf_record=rmldatatfrecord.core:tf_record
    '''
)
