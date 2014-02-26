# Distributions

WARNING
This is the unstable 2.0 branch of distributions,
which is a complete rewrite of the stable 1.0 master branch,
and which breaks API compatibility.

This package implements a variety of conjugate component models for
Bayesian MCMC inference.
Each model tries to have three implementations:
* a pure python implementation for correctness auditing
* a cython implementation for faster inference in python
* a low-precision C++ implementation for fastest inference in C++
although some models are missing some implementations.


## Cython

Distributions includes several optimized modules that require Cython
0.15.1 or later. On Ubuntu 12.04, this can be installed as follows:

    sudo apt-get install cython

After making a change to a .pyx or .pxd file, rebuild the extension:

    python setup.py build_ext --inplace

To install without cython:

    python setup.py --without-cython install

Or:

    pip install --global-option --without-cython .


## Component Model Interface

Component models are written as free functions with all data passed in
to emphasize their exact dependencies. The
distributions.ComponentModel interface class wraps these functions for
all models so that they may be used conveniently elsewhere.

The component model functions are strictly divided in to three responsibilities:

* mutating state

* scoring functions

* sampling functions

State change functions should be simple and fast.
Sampling functions consume explicit entropy sources.
Scoring functions may also consume entropy, when implemented with monte carlo.

Throughout the interface we use three types of data:

* model: global model state including fixed parameters and hyperparameters

* group: local component state including sufficient statistics and
  possibly group parameters

* value: individual observation state


## License

This code is released under the Revised BSD License.
See [LICENSE.txt](LICENSE.txt) for details.
