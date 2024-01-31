from setuptools import setup, find_packages
from os import remove
from json import dump
from pathlib import Path
from ravenml.utils.git import is_repo, git_sha, git_patch_tracked, git_patch_untracked

# figured out to use find_packages() via:
# https://stackoverflow.com/questions/10924885/is-it-possible-to-include-subdirectories-using-dist-utils-setup-py-as-part-of

pkg_name = 'rmldatatfrecord'

# attempt to write git data to file
# NOTE: does NOT work in the GitHub tarball installation case
# this will work in 3/4 install cases:
#   1. PyPI
#   2. GitHub clone
#   3. Local (editable), however NOTE in this case there is no need
#       for the file, as ravenml will find git information at runtime
#       in order to include patch data
plugin_dir = Path(__file__).resolve().parent
repo_root = is_repo(plugin_dir)
if repo_root:
    info = {
        'plugin_git_sha': git_sha(repo_root),
        'plugin_tracked_git_patch': git_patch_tracked(repo_root),
        'plugin_untracked_git_patch': git_patch_untracked(repo_root)
    }
    with open(plugin_dir / pkg_name / 'git_info.json', 'w') as f:
        dump(info, f, indent=2)

setup(
    name=pkg_name,
    version='0.1',
    description='Dataset creation plugin for ravenML',
    packages=find_packages(),
    package_data={pkg_name: ['git_info.json']},
    setup_requires=['ravenml'],
    install_requires=[
        'opencv-python',
        'numpy<1.19',
        'tensorflow~=2.3.0',
        'tqdm',
        'pandas',
    ],
    entry_points=f'''
        [ravenml.plugins.data]
        tf_record={pkg_name}.core:tf_record
    '''
)

# destroy git file after install
# NOTE: this is pointless for GitHub clone case, since the clone is deleted
# after install. It is necessary for local (editable) installs to prevent
# the file from corrupting the git repo, and when creating a dist for PyPI 
# for the same reason.
if repo_root:
    remove(plugin_dir / pkg_name / 'git_info.json')
