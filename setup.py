# with the help of setup.py i can build my entire machine learning project as a package and deploy in pypi
# It doesn’t run your code, it just provides instructions for installation.

from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name='mlproject',
    version='0.1.0',
    author='Danish Jain',
    author_email='danishsjain@gmail.com',
    packages =find_packages(),# finds src/ automatically
    install_requires=get_requirements('requirements.txt')
)