# Distributions [![Build Status](https://travis-ci.org/forcedotcom/distributions.png)](https://travis-ci.org/forcedotcom/distributions)

<b>WARNING</b>
This is the unstable 2.0 branch of distributions,
which is a complete rewrite of the stable 1.0 master branch,
and which breaks API compatibility.

This package implements basic building blocks for Bayesian MCMC inference
in Python and C++ including:
*   special numerical functions `distributions.<flavor>.special`,
*   samplers and density functions from a variety of distributions,
    `distributions.<flavor>.random`
*   conjugate component models (e.g., gamma-Poisson, normal-inverse-chi-squared)
    `distributions.<flavor>.models`, and.
*   clustering models (e.g., CRP, Pitman-Yor)
    `distributions.<flavor>.clustering`.

Python implementations are provided in up to three flavors:

*   Debug `distributions.dbg`
    are pure-python implementations for correctness auditing and
    error checking, and allowing debugging via pdb.

*   High-Precision `distributions.hp`
    are cython implementations for fast inference in python
    and numerical reference.

*   Low-Precision `distributions.lp`
    are inefficent wrappers of blazingly fast C++ implementations,
    intended mostly as wrappers to check that C++ implementations are correct.

Our typical workflow is to first prototype models in python,
then prototype faster inference applications using cython models,
and finally implement optimized scalable inference products in C++,
while testing all implementations for correctness.


## Installing

To build and install C++, Cython, and Python libraries into a virtualenv, simply

    make install

Then point `LD_LIBRARY_PATH` to find `libdistributions_shared.so`
in your virtualenv

    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate

Finally, test your installation with

    make test


### Cython

Distributions includes several optimized modules that require Cython
0.20.1 or later, as will automatically be installed by pip

To install without cython:

    python setup.py --without-cython install

or via pip:

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

        model.group_create(values=[]) -> group         # python only
        model.group_init(group)
        model.group_add_value(group, value)
        model.group_remove_value(group, value)
        model.group_merge(destin_group, source_group)
        model.plus_group(group) -> model               # optional

*   Sampling functions (optional in python).
    These consume explicit entropy sources in C++ or `global_rng` in python.

        model.sampler_init(sampler, group)            # c++ only
        model.sampler_create(group=empty) -> sampler  # python only, optional
        model.sampler_eval(sampler) -> value          # python only, optional
        model.sample_value(group) -> value
        model.sample_group(group_size) -> group

*   Scoring functions (optional in python).
    These may also consume entropy,
    e.g. when implemented using monte carlo integration)

        model.scorer_init(scorer, group)            # c++ only
        model.scorer_create(group=empty) -> scorer  # python only, optional
        model.scorer_eval(scorer, value) -> float   # python only, optional
        model.score_value(group, value) -> float
        model.score_group(group) -> float

*   Serialization to JSON (python only).

        model.load(json)
        model.dump() -> json
        group.load(json)
        group.dump() -> json
        ExampleModel.model_load(json) -> model
        ExampleModel.model_dump(model) -> json
        ExampleModel.group_load(json) -> group
        ExampleModel.group_dump(group) -> json

*   Testing metadata (python only).
    Example model parameters and datasets are automatically discovered by
    unit test infrastructures, reducing the cost of per-model test-writing.

        ExampleModel.EXAMPLES = [
            {'model': ..., 'values': [...]},
            ...
        ]


### Source of Entropy

The C++ methods explicity require a random number generator `rng` everywhere
entropy may be consumed.
The python models try to maintain compatibility with `numpy.random`
by hiding this source either as the global `numpy.random` generator,
or as single `global_rng` in wrapped C++.


## Authors (alphabetically)

* Beau Cronin <https://twitter.com/beaucronin>
* Jonathan Glidden <https://twitter.com/jhglidden>
* Eric Jonas <https://twitter.com/stochastician>
* Fritz Obermeyer <https://github.com/fritzo>
* Cap Petschulat <https://github.com/cap>


## License

Copyright 2014 Salesforce.com.

Licensed under the Revised BSD License.
See [LICENSE.txt](LICENSE.txt) for details.
