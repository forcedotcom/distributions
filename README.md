# Distributions [![Build Status](https://travis-ci.org/forcedotcom/distributions.png)](https://travis-ci.org/forcedotcom/distributions)

<b>WARNING</b>
This is the unstable 2.0 branch of distributions,
which is a complete rewrite of the stable 1.0 master branch,
and which breaks API compatibility.

This package implements a variety of conjugate component models for
Bayesian MCMC inference.
Each model may have up to three types of implementations:

*   `example_py` -
    a pure python implementation for correctness auditing and debugging.

*   `example_cy` -
    cython implementation for faster inference in python and debugging.

*   `example_cc` -
    a low-precision C++ implementation for fastest inference in C++.
    C++ models are available in the `include/` and `src/` directories,
    while their python wrappers enable unit testing.


## Cython

Distributions includes several optimized modules that require Cython
0.20.1 or later, as available via

    pip install cython

To install without cython:

    python setup.py --without-cython install

Or:

    pip install --global-option --without-cython .


## Component Model API

Component models are written as `Model` classes and `model.method` methods,
since all operations depend on the model.

Component models are written as free functions with all data passed in
to emphasize their exact dependencies. The
distributions.ComponentModel interface class wraps these functions for
all models so that they may be used conveniently elsewhere.

Each component model API consist of:

*   Datatypes.
    *   `ExampleModel` - global model state including fixed parameters
        and hyperparameters
    *   `ExampleModel.Value` - observation observation state
    *   `ExampleModel.Group` - local component state including
        sufficient statistics and possibly group parameters
    *   `ExampleModel.Sampler` -
        partially evaluated per-component sampling function
        (optional in python)
    *   `ExampleModel.Scorer` -
        partially evaluated per-component scoring function
        (optional in python)

*   State mutating functions.
    These should be simple and fast.

        model.group_init(group)
        model.group_add_value(group, value)
        model.group_remove_value(group, value)
        model.group_merge(destin_group, source_group)

*   Sampling functions. (optional in python)
    These consume explicit entropy sources in C++ or `global_rng` in python.

        model.sample_value(group) -> value
        model.sample_group(group_size) -> value

*   Scoring functions. (optional in python)
    These may also consume entropy,
    e.g. when implemented using monte carlo integration)

        model.sample_value(group, value) -> float
        model.sample_group(group) -> float

*   Serialization to JSON (python only).

        model.load(json)
        model.dump() -> json
        group.load(json)
        group.dump() -> json
        ExampleModel.load_model(json) -> model
        ExampleModel.dump_model(model) -> json
        ExampleModel.load_group(json) -> group
        ExampleModel.dump_group(group) -> json


## Source of Entropy

The C++ models explicity require a random number generator `rng` everywhere
entropy is consumed.
In python models, a single `global_rng` source is shared.


## License

This code is released under the Revised BSD License.
See [LICENSE.txt](LICENSE.txt) for details.
