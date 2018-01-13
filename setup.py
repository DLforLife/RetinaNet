#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    setup_requires=['pbr>=1.3', 'setuptools>=17.1',tensorflow>=1.2],
    packages=find_packages(),
    author="DLforLife",
    author_email="contact@dl4life.com",
    description="This is an implementation of RetinaNet ",
    license="MIT",
    keywords="tensorflow keras retinanet",
    url="https://github.com/DLforLife/RetinaNet",
    project_urls={
        "Bug Tracker": "https://github.com/DLforLife/RetinaNet",
        "Documentation": "https://github.com/DLforLife/RetinaNet",
        "Source Code": "https://github.com/DLforLife/RetinaNet/",
    }
    pbr=True)
