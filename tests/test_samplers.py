# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
The intention of this module is to test
the statistical correctness of samplers.

To compare empirical predictions with theoretical distributions
we use a (truncated) chi-squared test
applied to the TOPN most probable outcomes.
To apply the test to continuous distributions (NICH)
we bin samples based on their empirical cdf.
We also check that our (possibly binned) predictions assume
(1 - SUPPORT_THRESH)% of the probability mass.
This helps detect the case missed by the truncated
chi-squared test when we are correctly sampling
in a probable region but fail to sample improbable regions
i.e. never predicting OTHER in DPM or only sampling positive values for NICH.

These tests strive to be as robust as possible
but since they are based on random sampling they are expected
to give false positives occasionally.

"""

from distributions import ComponentModel
from distributions.util import multinomial_goodness_of_fit as mgof
from distributions.util import bin_samples, histogram
import numpy as np
from nose.tools import *
from scipy.integrate import quad


SAMPS = 10000
THRESH = 1e-3
TOPN = 8

DATA_COUNTS = [0, 10, 1000]


def _check_discrete(cm):
    samples = cm.sample_data(SAMPS)
    counts = histogram(samples)
    probs = np.exp([cm.pred_prob(x) for x in range(max(samples) + 1)])
    assert_less(1 - sum(probs), THRESH)
    probs, counts = zip(*sorted(zip(probs, counts), reverse=True)[:TOPN])
    p = mgof(probs, counts, SAMPS, truncated=True)
    assert_greater(p, THRESH)


def check_dd(impl, data_count, D):
    data = histogram(np.random.randint(D, size=data_count), bin_count=D)
    cm = ComponentModel(
            impl,
            ss={'counts': data},
            p={'D': D})
    cm.realize_hp()
    _check_discrete(cm)


def test_dd():
    for impl in ['pydd', 'cydd']:
        for D in [8, 16, 32]:
            for data_count in DATA_COUNTS:
                yield check_dd, impl, data_count, D


def check_dpm(impl, data_count, beta0):
    data = histogram(np.random.randint(50, size=data_count))
    data = dict([(str(i), obs) for i, obs in enumerate(data)])
    betas = dict([(str(i), (1 - beta0) / len(data))
        for i, obs in enumerate(data)])
    hp = {
            'gamma': 1.,
            'alpha': 1.,
            'beta0': beta0,
            'betas': betas
         }
    ss = {'counts': data}
    cm = ComponentModel(
            impl,
            ss=ss,
            hp=hp)
    samples = cm.sample_data(SAMPS)
    counts = list(histogram([y for y in samples if y != -1]))
    probs = list(np.exp([cm.pred_prob(x) for x in range(max(samples) + 1)]))
    counts.append(len([y for y in samples if y == -1]))
    probs.append(np.exp(cm.pred_prob(-1)))
    assert_less(1 - sum(probs), THRESH)
    probs, counts = zip(*sorted(zip(probs, counts), reverse=True)[:TOPN])
    p = mgof(probs, counts, SAMPS, truncated=True)
    assert_greater(p, THRESH)


def test_dpm():
    for impl in ['pydpm', 'cydpm']:
        for data_count in DATA_COUNTS:
            if data_count == 0:
                #dpm is degenerate with no data
                continue
            for beta0 in [.01, 0.5, .99]:
                yield check_dpm, impl, data_count, beta0


def check_gp(impl, data_count, lam):
    data = np.random.poisson(lam, size=data_count)
    ss = {
            'n': data_count,
            'sum': np.sum(data),
            'log_prod': np.sum(np.log(data))
         }
    cm = ComponentModel(impl, ss=ss)
    _check_discrete(cm)


def test_gp():
    for impl in ['pygp', 'cygp']:
        for lam in [10, 50, 100]:
            for data_count in DATA_COUNTS:
                yield check_gp, impl, data_count, lam


def check_nich(impl, data_count, mean, std):
    ss = None
    if data_count:
        data = np.random.normal(mean, std, size=data_count)
        ss = {
                'count': data_count,
                'mean': data.mean(),
                'variance': data.var()
             }
    cm = ComponentModel(impl, ss=ss)
    samples = cm.sample_data(SAMPS)
    counts, bin_ranges = bin_samples(samples)
    #use of quadrature is unfortunate but for now
    #it's the easiest way to score bins and seems to work
    pdf = lambda x: np.exp(cm.pred_prob(x))
    probs = [quad(pdf, m, M, epsabs=0., epsrel=1e-6)[0] for m, M in bin_ranges]
    assert_less(1 - sum(probs), THRESH)
    probs, counts = zip(*sorted(zip(probs, counts), reverse=True)[:TOPN])
    p = mgof(probs, counts, SAMPS, truncated=True)
    assert_greater(p, THRESH)


def test_nich():
    std = 5.
    for impl in ['pynich', 'cynich']:
        for mean in [0., -10., 10.]:
            for data_count in DATA_COUNTS:
                yield check_nich, impl, data_count, mean, std
