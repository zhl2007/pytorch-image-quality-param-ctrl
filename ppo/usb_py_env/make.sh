#!/bin/sh

 rm build -rf
 python3.6 setup.py build

 rm ../build -rf
 cp build ../ -rf
