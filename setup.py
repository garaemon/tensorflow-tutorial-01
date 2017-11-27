#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of tensorflow-tutorial-01.
# https://github.com/someuser/somepackage

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2017, garaemon <garaemon@gmail.com>

from setuptools import setup, find_packages
from tensorflow_tutorial_01 import __version__

tests_require = [
    'tensorflow',
]

setup(
    name='tensorflow-tutorial-01',
    version=__version__,
    description='tensorflow tutorial 01',
    long_description='''
tensorflow tutorial 01
''',
    keywords='tensorflow, ml, machine learning',
    author='garaemon',
    author_email='garaemon@gmail.com',
    url='https://github.com/someuser/somepackage',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        # add your dependencies here
        # remember to use 'package-name>=x.y.z,<x.y+1.0' notation (this way you get bugfixes)
    ],
    extras_require={
        'tests': tests_require,
    },
    entry_points={
        'console_scripts': [
            # add cli scripts here in this form:
            # 'tensorflow-tutorial-01=tensorflow_tutorial_01.cli:main',
        ],
    },
)
