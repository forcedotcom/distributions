#!/bin/bash

pip install -e . && \
pyflakes setup.py distributions && \
pep8 --repeat setup.py distributions && \
nosetests -v && \
./ctest.sh
