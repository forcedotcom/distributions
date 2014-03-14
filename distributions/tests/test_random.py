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

import itertools
import numpy
import scipy
from nose import SkipTest
from nose.tools import (
    assert_less,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)
from distributions.tests.util import assert_close, assert_samples_match_scores
from distributions.dbg.random import sample_stick, sample_discrete_log


SAMPLES = 1000


def assert_normal(x, y, sigma, stddevs=4.0):
    '''
    Assert that the difference between two values is within a few standard
    deviations of the predicted [normally distributed] error of zero.
    '''
    assert_less(x, y + sigma * stddevs)
    assert_less(y, x + sigma * stddevs)


def test_seed():
    try:
        import distributions.hp.random
    except ImportError:
        raise SkipTest('no cython support')
    global_rng = distributions.hp.random.random
    distributions.hp.random.seed(0)
    values1 = [global_rng() for _ in xrange(10)]
    distributions.hp.random.seed(0)
    values2 = [global_rng() for _ in xrange(10)]
    assert_equal(values1, values2)


def test_sample_discrete_log_underflow():
    sample_discrete_log([-1e3])
    sample_discrete_log([-1e3, -1e-3])


def test_sample_discrete_log():
    assert_equal(sample_discrete_log([-1.]), 0)
    assert_equal(sample_discrete_log([-1e3]), 0)
    assert_equal(sample_discrete_log([-1e-3]), 0)
    assert_equal(sample_discrete_log([-1., -1e3]), 0)
    assert_equal(sample_discrete_log([-1e3, -1.]), 1)
    assert_raises(Exception, sample_discrete_log, [])


def test_sample_stick():
    gammas = [.1, 1., 5., 10.]
    for gamma in gammas:
        for _ in range(5):
            betas = sample_stick(gamma).values()
            assert_almost_equal(sum(betas), 1., places=5)


def scipy_normal_draw(mean, variance):
    return scipy.stats.norm.rvs(mean, numpy.sqrt(variance))


def test_normal_draw():
    try:
        import distributions.hp.random
    except ImportError:
        raise SkipTest('no cython support')
    means = [1.0 * i for i in range(-2, 3)]
    variances = [10.0 ** i for i in range(-3, 4)]
    for mean, variance in itertools.product(means, variances):
        # Assume scipy.stats is correct
        #yield _test_normal_draw, scipy_normal_draw, mean, variance
        _test_normal_draw(
            distributions.hp.random.sample_normal,
            mean,
            variance)


def _test_normal_draw(draw, mean, variance):
    samples = [draw(mean, variance) for _ in range(SAMPLES)]
    assert_normal(numpy.mean(samples), mean, numpy.sqrt(variance / SAMPLES))
    error = numpy.array(samples) - mean
    chisq = numpy.dot(error, error) / variance
    assert_normal(chisq, SAMPLES, 2 * SAMPLES)


def test_chisq_draw():
    try:
        import distributions.hp.random
    except ImportError:
        raise SkipTest('no cython support')
    nus = [1.5 ** i for i in range(-10, 11)]
    for nu in nus:
        # Assume scipy.stats is correct
        #yield _test_chisq_draw, scipy.stats.chi2.rvs, nu
        _test_chisq_draw(distributions.hp.random.sample_chisq, nu)


def _test_chisq_draw(draw, nu):
    samples = [draw(nu) for _ in range(SAMPLES)]
    assert_normal(numpy.mean(samples), nu, numpy.sqrt(2 * nu / SAMPLES))


def test_sample_pair_from_urn():
    try:
        import distributions.lp.random
    except ImportError:
        raise SkipTest('no cython support')
    TEST_FAIL_PROB = 1e-5
    ITEM_COUNT = 10

    items = range(ITEM_COUNT)
    counts = {(i, j): 0 for i in items for j in items if i != j}
    pair_count = len(counts)

    def test_fail_prob(sample_count):
        '''
        Let X1,...,XK ~iid uniform({1, ..., N = pair_count})
        and for n in {1,..,N} let Cn = sum_k (1 if Xk = n else 0).
        Then for each n,

            P(Cn = 0) = ((N-1) / N)^K
            P(Cn > 0) = 1 - ((N-1) / N)^K
            P(test fails) = 1 - P(for all n, Cn > 0)
                          ~ 1 - (1 - ((N-1) / N)^K)^N
        '''
        item_fail_prob = ((pair_count - 1.0) / pair_count) ** sample_count
        test_fail_prob = 1 - (1 - item_fail_prob) ** pair_count
        return test_fail_prob

    sample_count = 1
    while test_fail_prob(sample_count) > TEST_FAIL_PROB:
        sample_count *= 2
    print 'pair_count = {}'.format(pair_count)
    print 'sample_count = {}'.format(sample_count)

    for _ in xrange(sample_count):
        i, j = distributions.lp.random.sample_pair_from_urn(items)
        assert i != j
        counts[i, j] += 1

    assert_less(0, min(counts.itervalues()))


def test_prob_from_scores():
    try:
        import distributions.lp.random
    except ImportError:
        raise SkipTest('no cython support')
    rng1 = distributions.lp.random.RNG(0)
    rng2 = distributions.lp.random.RNG(0)
    for size in range(1, 100):
        scores = numpy.random.normal(size=size).tolist()
        for _ in xrange(size):
            sample, prob1 = distributions.lp.random.sample_prob_from_scores(
                rng1,
                scores)
            assert 0 <= sample and sample < size
            prob2 = distributions.lp.random.prob_from_scores(
                rng2,
                sample,
                scores)
            assert_close(
                prob1,
                prob2,
                err_msg='sample_prob_from_scores != prob_from_scores')


def test_sample_prob_from_scores():
    try:
        import distributions.lp.random
    except ImportError:
        raise SkipTest('no cython support')
    rng = distributions.lp.random.RNG(0)
    for size in range(1, 10):
        scores = numpy.random.normal(size=size).tolist()

        def sampler():
            return distributions.lp.random.sample_prob_from_scores(rng, scores)

        assert_samples_match_scores(sampler)
