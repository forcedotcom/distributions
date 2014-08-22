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

Install numpy and scipy. Then::

    pip install distributions


C++ Standalone
--------------

Install requirements::

    sudo apt-get install cmake libeigen3-dev

To install in ``./lib``::

    make install

Alternatively, set a custom install location::

    CMAKE_INSTALL_PREFIX=/my/prefix make install


Python wrapping libdistributions
--------------------------------

Follow instructions for C++ Standalone. Install numpy and scipy. Then::

    LIBRARY_PATH=/my/prefix/lib pip install distributions

.. warning::

    When using wrapped libdistributions, the dynamic linker must be
    able to find the library. The environment variables used to do
    this differ from platform to platform.

    On Linux, you might run python as follows::

        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/prefix/lib python

    On OSX, you'll need a different flag::

        DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/my/prefix/lib python

    If you use virtualenv with virtualenvwrapper and use the
    virtualenv root as your prefix, it is convenient to add a
    postactivate hook to set this environment. On Linux, this would
    look like this::

        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate

Developer Quick Start
---------------------

This will install both the static and dynamic versions of
libdistributions within a virtualenv, then install the distributions
Python package built to wrap libdistributions.

Install cmake. Install numpy, scipy, cython, and nosetests so that
they're available within a python virtualenv. Activate that
virtualenv. Then::

    make test

The top-level ``Makefile`` provides many targets useful for development.
