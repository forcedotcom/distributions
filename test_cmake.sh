#!/bin/bash

BUILD=examples/cmake/build
ROOT=$PWD

mkdir -p $BUILD
cd $BUILD

CMAKE_PREFIX_PATH=${VIRTUAL_ENV} cmake .. && make && ./foo
