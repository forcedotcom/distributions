#!/bin/bash

BUILD=examples/cmake/build
ROOT=$PWD

mkdir -p $BUILD
cd $BUILD

echo '------------'
echo 'REMOTE BUILD'
cmake .. && make && ./foo

echo '-----------'
echo 'LOCAL BUILD'
DISTRIBUTIONS_PATH=$ROOT cmake .. && make && ./foo
