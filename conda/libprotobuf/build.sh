#!/bin/sh
export CXXFLAGS="-I${PREFIX}/include" 
export LDFLAGS="-L${PREFIX}/lib"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig"
if [ "`uname`" = "Darwin" ]; then
    # conda sets MACOSX_DEPLOYMENT_TARGET=10.5, but we need to build with
    # -stdlib=libc++; see
    # https://code.google.com/p/protobuf/issues/detail?id=449
    export MACOSX_DEPLOYMENT_TARGET=10.7
    export CXXFLAGS="${CXXFLAGS}"
    export CC=clang
    export CXX="clang++ -std=c++11 -stdlib=libc++"
fi
./configure --prefix=${PREFIX} --without-zlib
make -j
make install
