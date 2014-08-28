[![Build Status](https://travis-ci.org/forcedotcom/distributions.svg?branch=master)](https://travis-ci.org/forcedotcom/distributions)
[![Code Quality](http://img.shields.io/scrutinizer/g/forcedotcom/distributions.svg)](https://scrutinizer-ci.com/g/forcedotcom/distributions/code-structure/master/hot-spots)
[![Latest Version](https://badge.fury.io/py/distributions.svg)](https://pypi.python.org/pypi/distributions)

# Distributions

Distributions provides low-level primitives for
collapsed Gibbs sampling in Python and C++ including:

* special numerical functions,
* samplers and density functions from a variety of distributions,
* conjugate component models (e.g., gamma-Poisson, normal-inverse-chi-squared),
* clustering models (e.g., CRP, Pitman-Yor), and
* efficient wrappers for mixture models.

Distributions powered a machine-learning-as-a-service for Prior Knowledge Inc.,
and now powers machine learning infrastructure at Salesforce.com.


## Installation

For python-only support (no C++) you can install with pip:

    pip install distributions

For help with other builds, see
[the installation documentation](http://distributions.readthedocs.org/en/latest/installation.html).


## Documentation

The official documentation lives at http://distributions.readthedocs.org/.

Branch-specific documentation lives at

* [Overview](/doc/overview.rst)
* [Installation](/doc/installation.rst)


## Authors (alphabetically)

* Jonathan Glidden <https://twitter.com/jhglidden>
* Eric Jonas <https://twitter.com/stochastician>
* Fritz Obermeyer <https://github.com/fritzo>
* Cap Petschulat <https://github.com/cap>
* Stephen Tu <https://github.com/stephentu>

## License

Copyright (c) 2014 Salesforce.com, Inc. All rights reserved.

Licensed under the Revised BSD License. See [LICENSE.txt](LICENSE.txt)
for details.
