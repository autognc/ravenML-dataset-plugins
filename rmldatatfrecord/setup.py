from setuptools import setup, find_packages
from os import remove
from json import dump
from shutil import copyfile
from pathlib import Path
from ravenml.utils.git import is_repo, git_sha, git_patch_tracked, git_patch_untracked

# figured out to use find_packages() via:
# https://stackoverflow.com/questions/10924885/is-it-possible-to-include-subdirectories-using-dist-utils-setup-py-as-part-of

# determine GPU or CPU install via env variable
# gpu = os.getenv('RML_BBOX_GPU')
# tensorflow_pkg = 'tensorflow==1.14.0' if not gpu else 'tensorflow-gpu==1.14.0'

plugin_dir = Path(__file__).resolve().parent

# attempt to write git data to file
# this will work in 3/4 install cases:
#   1. PyPI
#   2. GitHub clone
#   3. Local (editable), NOTE in this case there is no need
#       for the file, as ravenml will find git information at runtime
# NOTE: does NOT work in the GitHub tarball installation case
repo = is_repo(plugin_dir)
if repo:
    info = {
        'plugin_git_sha': git_sha(plugin_dir),
        'plugin_tracked_git_patch': git_patch_tracked(plugin_dir),
        'plugin_untracked_git_patch': git_patch_untracked(plugin_dir)
    }
    with open(plugin_dir / 'rmldatatfrecord' / 'git_info.json', 'w') as f:
        dump(info, f, indent=2)

setup(
    name='rmldatatfrecord',
    version='0.1',
    description='Dataset creation plugin for ravenML',
    packages=find_packages(),
    package_data={'rmldatatfrecord': ['git_info.json']},
    setup_requires=['ravenml'],
    install_requires=[
        'opencv-python',
        'numpy<1.19',
        'tensorflow>=2.3',
        'tqdm',
    ],
    entry_points='''
        [ravenml.plugins.data]
        tf_record=rmldatatfrecord.core:tf_record
    '''
)

# destroy git file after install
# NOTE: this is pointless for GitHub clone case, since the clone is deleted
# after install. It is necessary for local (editable) installs to prevent
# the file from corrupting the git repo, and when creating a dist for PyPI 
# for the same reason.
if repo:
    remove(plugin_dir / 'rmldatatfrecord' / 'git_info.json')
