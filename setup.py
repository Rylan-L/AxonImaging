# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 17:25:19 2018

@author: rylanl
"""

from distutils.core import setup

setup(
    name='axonimaging',
    version='0.1.0',
    author='Rylan Larsen',
    author_email='RylanL@alleninstitute.org',
    packages=['AxonImaging'],
    url='https://github.com/Rylan-L/AxonImaging',
    license='LICENSE',
    description='Signal and Image Processing with an Emphasis on Neurophysiology data',
    long_description=open('README.md').read(),
    install_requires=[
        "Tifffile >= 0.10.0",
    
    ],
)