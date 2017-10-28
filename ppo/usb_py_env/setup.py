#!/usr/bin/env python

"""
python2.7 setup.py build
"""

from distutils.core import setup, Extension

MOD = 'UsbCamEnv'
setup(name=MOD, ext_modules=[Extension(MOD, sources=['usb_cam_env.c'])])