from setuptools import setup, find_packages

# Project metadata
NAME = 'sam-loss-landscape'
DESCRIPTION = 'A pipeline for visualizing loss landscape with optimzier SAM'
VERSION = '0.1'
AUTHOR = 'Hoang-Chau Luong'
AUTHOR_EMAIL = 'lhchau20@apcs.fitus.edu.vn'
URL = 'https://github.com/lhchau'

# Specify project dependencies
INSTALL_REQUIRES = [
    # List your dependencies here, e.g., 'numpy>=1.0'
]

# Define the setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)
