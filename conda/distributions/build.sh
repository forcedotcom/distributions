#!/bin/sh
echo "Conda build env"
printenv
echo "protoc: `which protoc`"
# XXX(stephentu): replace with git tag once this is all in place
SHA1=e1c80d04bb94a2af2f6384066a2c1209869a04ff
git clone https://github.com/forcedotcom/distributions.git
cd distributions && git checkout ${SHA1}
make protobuf
mkdir build && cd build
DISTRIBUTIONS_USE_PROTOBUF=1 cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DEXTRA_INCLUDE_PATH=${PREFIX}/include \
    -DEXTRA_LIBRARY_PATH=${PREFIX}/lib ..
make VERBOSE=1 && make install
cd ..
PYDISTRIBUTIONS_USE_LIB=1 LIBRARY_PATH=${PREFIX}/lib EXTRA_INCLUDE_PATH=${PREFIX}/include $PYTHON setup.py install 
