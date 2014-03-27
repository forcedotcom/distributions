Installation
============

You may build distributions in several ways:

* as a standalone C++ library
* as a standalone Python package
* as a Python package wrapping the dynamically-linked C++ library

.. note::

    On OSX, distributions builds with newer versions of clang, but
    some systems default to g++. You can force distributions to use
    clang by setting the ``CC`` environment variable before running
    any ``pip``, ``cmake``, or ``make`` commands with ``export
    CC=clang``.


Python Standalone
-----------------

COMING SOON: Install numpy and scipy. Then::

    pip install distributions


C++ Standalone
--------------

Install cmake. Then::

    mkdir build; cd build
    cmake -DCMAKE_INSTALL_PREFIX=/my/prefix ..
    make install


Python wrapping libdistributions
--------------------------------

Follow instructions for C++ Standalone. Install numpy and scipy. Then::

    PYDISTRIBUTIONS_USE_LIB=1 LIBRARY_PATH=/my/prefix/lib pip install distributions


Developer Quick Start
---------------------

This will install both the static and dynamic versions of
libdistributions within a virtualenv, then install the distributions
Python package built to wrap libdistributions.

Install cmake. Install numpy, scipy, cython, and nosetests so that
they're available within a python virtualenv. Activate that
virtualenv. Then::

    PYDISTRIBUTIONS_USE_LIB=1 make test

The top-level ``Makefile`` provides many targets useful for
development. The command above will:
