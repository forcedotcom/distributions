#!/bin/sh
if [ "`uname`" = "Darwin" ]; then
    export MACOSX_DEPLOYMENT_TARGET=10.7
fi
echo "Conda build env"
printenv
echo "protoc: `which protoc`"
git clone https://github.com/posterior/distributions.git
cd distributions && git checkout tags/2.0.26
make protobuf
mkdir build && cd build
DISTRIBUTIONS_USE_PROTOBUF=1 cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DEXTRA_INCLUDE_PATH=${PREFIX}/include \
    -DEXTRA_LIBRARY_PATH=${PREFIX}/lib ..
make VERBOSE=1 && make install
cd ..
LIBRARY_PATH=${PREFIX}/lib \
    EXTRA_INCLUDE_PATH=${PREFIX}/include \
    $PYTHON setup.py install 
