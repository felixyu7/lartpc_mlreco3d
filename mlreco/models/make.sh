#!/usr/bin/env bash


python3 setup_bbox.py build_ext --inplace --user
rm -rf build

python3 setup_layers.py build develop --user

