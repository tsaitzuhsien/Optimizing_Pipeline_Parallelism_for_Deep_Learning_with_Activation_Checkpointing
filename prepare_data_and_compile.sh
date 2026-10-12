#!/bin/bash

mkdir -p profile_dicts

cd data/openwebtext
python prepare.py

cd ../../algorithm
./pybind_compile.sh
