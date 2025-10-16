from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT='-e .'


def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[] # create an empty list
    with open(file_path) as file_obj: # open the txt file as file_obj
        requirements=file_obj.readlines() # save it to the reqi=uirements list [ ]
        requirements=[req.replace("\n","") for req in requirements] # replace the next line with blank(error free)
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT) # remove the '-e .'
            
    return requirements # return the list


setup(
    name='Raahi_Trekkers',
    version='0.0.1',
    author="Pratap Jadhav",
    author_email="pratap.jadhav0@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)