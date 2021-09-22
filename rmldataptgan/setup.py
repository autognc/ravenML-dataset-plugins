from setuptools import setup, find_packages
from os import remove
from json import dump
from pathlib import Path
from ravenml.utils.git import is_repo, git_sha, git_patch_tracked, git_patch_untracked

# figured out to use find_packages() via:
# https://stackoverflow.com/questions/10924885/is-it-possible-to-include-subdirectories-using-dist-utils-setup-py-as-part-of

pkg_name = 'rmldataptgan'



setup(
    name=pkg_name,
    version='0.1',
    description='Dataset creation plugin for ravenML',
    packages=find_packages(),
    setup_requires=['ravenml'],
    install_requires=[
        'numpy<1.19',
        'tqdm',
    ],
    entry_points=f'''
        [ravenml.plugins.data]
        pt_gan={pkg_name}.core:pt_gan
    '''
)
