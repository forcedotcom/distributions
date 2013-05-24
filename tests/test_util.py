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
from nose.tools import *

from distributions.util import (scores_to_probs,
        discrete_draw_log, stick, bin_samples)
from distributions.util import multinomial_goodness_of_fit as mgof


def test_discrete_draw_log_underflow():
    discrete_draw_log([-1e3])
    discrete_draw_log([-1e3, -1e-3])


def test_discrete_draw_log():
    assert_equal(discrete_draw_log([-1.]), 0)
    assert_equal(discrete_draw_log([-1e3]), 0)
    assert_equal(discrete_draw_log([-1e-3]), 0)
    assert_equal(discrete_draw_log([-1., -1e3]), 0)
    assert_equal(discrete_draw_log([-1e3, -1.]), 1)
    assert_raises(Exception, discrete_draw_log, [])


def test_stick():
    gammas = [.1, 1., 5., 10.]
    for gamma in gammas:
        for _ in range(5):
            betas = stick(gamma).values()
            assert_almost_equal(sum(betas), 1., places=5)


def test_scores_to_probs():
    scores = [-10000, 10000, 10001, 9999, 0, 5, 6, 6, 7]
    probs = scores_to_probs(scores)
    assert_less(abs(sum(probs) - 1), 1e-6)
    for prob in probs:
        assert_less_equal(0, prob)
        assert_less_equal(prob, 1)


def test_multinmoial_goodness_of_fit():
    thresh = 1e-3
    n = int(1e5)
    ds = [3, 10, 20]
    for d in ds:
        for _ in range(5):
            probs = np.random.dirichlet([1] * d)
            counts = np.random.multinomial(n, probs)
            p_good = mgof(probs, counts, n)
            assert_greater(p_good, thresh)

        unif_counts = np.random.multinomial(n, [1. / d] * d)
        p_bad = mgof(probs, unif_counts, n)
        assert_less(p_bad, thresh)


def test_bin_samples():
    samples = range(6)
    np.random.shuffle(samples)
    counts, bounds = bin_samples(samples, 2)
    assert_list_equal(list(counts), [3, 3])
    assert_list_equal(list(bounds[0]), [0, 3])
    assert_list_equal(list(bounds[1]), [3, 5])
