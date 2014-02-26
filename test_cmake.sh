#!/bin/bash

BUILD=examples/cmake/build

mkdir -p $BUILD
cd $BUILD
cmake ..
make
./foo
