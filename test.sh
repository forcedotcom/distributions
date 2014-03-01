#!/bin/bash

pyflakes setup.py distributions && \
pep8 --repeat setup.py distributions && \
nosetests -v && \
./ctest.sh
