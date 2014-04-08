#!/usr/bin/env python


import os
import sys
from distutils.spawn import find_executable
from distutils import sysconfig
from setuptools import setup

os.environ['CXX'] = 'nvcc'
os.environ['CC'] = 'nvcc'

def log(str):
  print >>sys.stderr, str

if find_executable('nvcc') is None:
  log('nvcc not in path; aborting')
  sys.exit(1)

log('About to build cudaconv2 extension.')
cmd = 'cd cudaconv2 && make PYTHON_INCLUDE="%s"' % sysconfig.get_python_inc()

log(cmd)
if os.system(cmd) != 0:
  log('Failed to build extension')
  sys.exit(1)


extension_modules = [] 

setup(
    name="fastnet",
    description="Fast convolution network library",
    long_description='',
    author="Russell Power & Justin Lin",
    author_email="power@cs.nyu.edu",
    license="GPL",
    version="0.1",
    url="http://github.com/rjpower/fastnet",
    packages=[ 'fastnet', 'cudaconv2', ],
    package_dir={ 
      'fastnet' : 'fastnet',
      'cudaconv2' : 'cudaconv2' ,
    },
    data_files = [
     ('cudaconv2',  ['cudaconv2/_cudaconv2.so'])
    ],
    install_requires=[
      'pycuda', 
      'numpy',
      'traits',
      'scikits.cuda',
    ],
    ext_modules = extension_modules)
