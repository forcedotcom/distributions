# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
from nose.tools import (
    assert_less,
    assert_less_equal,
    assert_greater,
    assert_list_equal,
)
from distributions.util import (
    scores_to_probs,
    bin_samples,
    multinomial_goodness_of_fit,
)
from distributions.tests.util import seed_all


def test_scores_to_probs():
    scores = [-10000, 10000, 10001, 9999, 0, 5, 6, 6, 7]
    probs = scores_to_probs(scores)
    assert_less(abs(sum(probs) - 1), 1e-6)
    for prob in probs:
        assert_less_equal(0, prob)
        assert_less_equal(prob, 1)


def test_multinomial_goodness_of_fit():
    for dim in range(2, 20):
        yield _test_multinomial_goodness_of_fit, dim


def _test_multinomial_goodness_of_fit(dim):
    seed_all(0)
    thresh = 1e-3
    sample_count = int(1e5)
    probs = numpy.random.dirichlet([1] * dim)

    counts = numpy.random.multinomial(sample_count, probs)
    p_good = multinomial_goodness_of_fit(probs, counts, sample_count)
    assert_greater(p_good, thresh)

    unif_counts = numpy.random.multinomial(sample_count, [1. / dim] * dim)
    p_bad = multinomial_goodness_of_fit(probs, unif_counts, sample_count)
    assert_less(p_bad, thresh)


def test_bin_samples():
    samples = range(6)
    numpy.random.shuffle(samples)
    counts, bounds = bin_samples(samples, 2)
    assert_list_equal(list(counts), [3, 3])
    assert_list_equal(list(bounds[0]), [0, 3])
    assert_list_equal(list(bounds[1]), [3, 5])
