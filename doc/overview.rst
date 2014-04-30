Overview
========

Distributions implements low-level primitives for Bayesian MCMC
inference in Python and C++ including:

* special numerical functions ``distributions.<flavor>.special``,

* samplers and density functions from a variety of distributions,
  ``distributions.<flavor>.random``,

* conjugate component models (e.g., gamma-Poisson,
  normal-inverse-chi-squared) ``distributions.<flavor>.models``, and

* clustering models (e.g., CRP, Pitman-Yor)
  ``distributions.<flavor>.clustering``.

Python implementations are provided in up to three flavors:

* Debug ``distributions.dbg`` are pure-python implementations for
  correctness auditing and error checking, and allowing debugging via
  pdb.

* High-Precision ``distributions.hp`` are cython implementations for
  fast inference in python and numerical reference.

* Low-Precision ``distributions.lp`` are inefficent wrappers of
  blazingly fast C++ implementations, intended mostly as wrappers to
  check that C++ implementations are correct.

Our typical workflow is to first prototype models in python,
then prototype faster inference applications using cython models,
and finally implement optimized scalable inference products in C++,
while testing all implementations for correctness.


Component Model API
-------------------

Component models are written as ``Model`` classes and ``model.method`` methods,
since all operations depend on the model.

Component models are written as free functions with all data passed in
to emphasize their exact dependencies. The
distributions.ComponentModel interface class wraps these functions for
all models so that they may be used conveniently elsewhere.

Each component model API consist of:

*   Datatypes.
    *   ``ExampleModel`` - global model state including fixed parameters
        and hyperparameters
    *   ``ExampleModel.Value`` - observation observation state
    *   ``ExampleModel.Group`` - local component state including
        sufficient statistics and possibly group parameters
    *   ``ExampleModel.Sampler`` -
        partially evaluated per-component sampling function
        (optional in python)
    *   ``ExampleModel.Scorer`` -
        partially evaluated per-component scoring function
        (optional in python)
    *   ``ExampleModel.Classifier`` - vectorized scoring functions
        (optional in python)

*   State mutating functions.
    These should be simple and fast::

        model.group_create(values=[]) -> group         # python only
        group.init(model)
        group.add_value(model, value)
        group.remove_value(model, value)
        group.merge(model, other_group)
        model.plus_group(group) -> model               # optional

*   Sampling functions (optional in python).
    These consume explicit entropy sources in C++ or ``global_rng`` in python::

        model.sampler_create(group=empty) -> sampler  # python only, optional
        sampler.init(model, group)                    # c++ only
        sampler.eval(sampler) -> value                # python only, optional
        model.sample_value(group) -> value
        model.sample_group(group_size) -> group

*   Scoring functions (optional in python).
    These may also consume entropy,
    e.g. when implemented using monte carlo integration)::

        model.scorer_create(group=empty) -> scorer  # python only, optional
        scorer.init(model, group)                   # c++ only
        scorer.eval(model, value) -> float          # python only, optional
        model.score_value(group, value) -> float
        model.score_group(group) -> float

*   Mixture slave (optional in python).
    These provide batch operations on a collection of groups.

    Clustering mixture drivers::

        mixture.sample_assignments(sample_size)
        mixture.score_counts(model)
        mixture.add_group(model)
        mixture.remove_group(model, groupid)
        mixture.add_value(model, groupid, value)
        mixture.remove_value(model, groupid, value)
        mixture.score_value(model, value, scores_accum)

    Component model mixture slaves::

        mixture.groups.push_back(group)          # c++ only
        mixture.append(group)                    # python only
        mixture.init(model)
        mixture.add_group(model)
        mixture.remove_group(model, groupid)
        mixture.add_value(model, groupid, value)
        mixture.remove_value(model, groupid, value)
        mixture.score_value(model, value, scores_accum)

*   Serialization to JSON (python only)::

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
    unit test infrastructures, reducing the cost of per-model test-writing::

        ExampleModel.EXAMPLES = [
            {'model': ..., 'values': [...]},
            ...
        ]


Clustering Model API
--------------------

*   Sampling and scoring::

        model.sample_assignments(sample_size)
        model.score_counts(counts)
        model.score_add_value(...)
        model.score_remove_value(...)

*   Mixture driver (optional in python).
    These provide batch operations on a collection of groups.::

        mixture.init(model, counts)
        mixture.add_value(model, groupid, count)
        mixture.remove_value(model, groupid, count)
        mixture.score(model, scores)

*   Testing metadata (python only).
    Example model parameters and datasets are automatically discovered by
    unit test infrastructures, reducing the cost of per-model test-writing::

        ExampleModel.EXAMPLES = [ ...model specific... ]


Source of Entropy
-----------------

The C++ methods explicity require a random number generator ``rng``
everywhere entropy may be consumed. The python models try to maintain
compatibility with ``numpy.random`` by hiding this source either as
the global ``numpy.random`` generator, or as single ``global_rng`` in
wrapped C++.
