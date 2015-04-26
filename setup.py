# -*- coding: utf-8 -*-
"""Build and install the windspharm package."""
from setuptools import setup

for line in open('lib/pyveccalc/__init__.py').readlines():
    if line.startswith('__version__'):
        exec(line)

packages = ['pyveccalc']

setup(name='pyveccalc',
      version=__version__,
      description='Wind vector calculations in finite differences',
      author='Denis Sergeev',
      author_email='d.sergeev@uea.ac.uk',
      url='https://github.com/dennissergeev/pyveccalc',
      long_description="""
      Python library for wind vector calculations in finite differences
      """,
      packages=packages,
      package_dir={'':'lib'},
      install_requires=['numpy'],)
