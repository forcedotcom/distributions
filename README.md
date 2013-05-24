# Distributions

Code in this package is meant to be understandable and auditable for
correctness. It is not meant to perform especially well.


## Cython

Distributions includes several optimized modules that require Cython
0.15.1 or later. On Ubuntu 12.04, this can be installed as follows:

    sudo apt-get install cython

After making a change to a .pyx or .pxd file, rebuild the extension:

    python setup.py build_ext --inplace


## Component Model Interface

Component models are written as free functions with all data passed in
to emphasize their exact dependencies. The
distributions.ComponentModel interface class wraps these functions for
all models so that they may be used conveniently elsewhere.

The component model functions are strictly divided in to two sets,
those which change state and those which do interesting math. State
change functions should be simple and fast.

Throughout the interface we use several short variable names:

* ss: sufficient statistics

* hp: hyperparameters

* p: parameters

* y: data value

The functions are as follows:

* `create_ss(ss=None, p=None)` Create a valid suff stats dict,
  optionally incorporating the suff stats and parameters provided in
  the arguments.

* `dump_ss(ss)` Return a serializable version of the sufficient
  statistics contained in the argument.

* `create_hp(hp=None, p=None)` Create a valid hyperparameters dict,
  optionally incorporating the hps and parameters provided in the
  argument.

* `dump_hp(hp)` Return a serializable version of the hyperparameters
  contained in the argument.

* `add_data(ss, y)` Add the datapoint to the suff stats. Does not
  explicitly check that the datapoint is valid. Returns None.

* `remove_data(ss, y)` Removethe datapoint from the suff stats. Does
  not explicitly check that the datapoint is valid, or that it was
  added previously. Returns None.

* `sample_data(hp, ss)` Sample a datapoint from the distribution
  described by the hyperparameters and suff stats.

* `sample_post(hp, ss)` Sample the model parameters from the posterior
  distribution described by the hyperparameters and suff stats.

* `pred_prob(hp, ss, y)` Compute the posterior predictive probability
  of the datapoint, given the hyperparameters and suff stats (and
  integrating over the model parameters).

* `data_prob(hp, ss)` Compute the marginal probability of all of the
  data summarized by the suff stats, given the hyperparameters (and
  integrating over the model parameters).
