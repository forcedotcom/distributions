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

import numpy as np
from distributions.conjugate import scimath
import itertools
from nose.tools import assert_less
import scipy


SAMPLES = 1000


def assert_close(x, y, sigma, stddevs=4.0):
    '''
    Assert that the difference between two values is within a few standard
    deviations of the predicted [normally distributed] error of zero.
    '''
    assert_less(x, y + sigma * stddevs)
    assert_less(y, x + sigma * stddevs)


def scipy_normal_draw(mean, variance):
    return scipy.stats.norm.rvs(mean, np.sqrt(variance))


def _test_normal_draw(draw, mean, variance):
    samples = [draw(mean, variance) for _ in range(SAMPLES)]
    assert_close(np.mean(samples), mean, np.sqrt(variance / SAMPLES))
    error = np.array(samples) - mean
    chisq = np.dot(error, error) / variance
    assert_close(chisq, SAMPLES, 2 * SAMPLES)


def test_normal_draw():
    means = [1.0 * i for i in range(-2, 3)]
    variances = [10.0 ** i for i in range(-3, 4)]
    for mean, variance in itertools.product(means, variances):
        # Assume scipy.stats is correct
        #yield _test_normal_draw, scipy_normal_draw, mean, variance
        yield _test_normal_draw, scimath.normal_draw, mean, variance


def _test_chisq_draw(draw, nu):
    samples = [draw(nu) for _ in range(SAMPLES)]
    assert_close(np.mean(samples), nu, np.sqrt(2 * nu / SAMPLES))


def test_chisq_draw():
    nus = [1.5 ** i for i in range(-10, 11)]
    for nu in nus:
        # Assume scipy.stats is correct
        #yield _test_chisq_draw, scipy.stats.chi2.rvs, nu
        yield _test_chisq_draw, scimath.chisq_draw, nu
