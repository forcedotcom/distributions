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
from nose.tools import (
    assert_less,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)
from distributions.dbg.random import (
    sample_stick,
    sample_discrete_log,
    sample_inverse_wishart,
    score_student_t as dbg_score_student_t,
)
from distributions.lp.random import (
    score_student_t as lp_score_student_t,
)
from distributions.dbg.models.nich import (
    score_student_t as scalar_score_student_t,
)
from distributions.tests.util import (
    require_cython,
    assert_close,
    assert_samples_match_scores,
    seed_all,
)


SAMPLES = 1000


def assert_normal(x, y, sigma, stddevs=4.0):
    '''
    Assert that the difference between two values is within a few standard
    deviations of the predicted [normally distributed] error of zero.
    '''
    assert_less(x, y + sigma * stddevs)
    assert_less(y, x + sigma * stddevs)


def test_seed():
    require_cython()
    import distributions.hp.random
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
    require_cython()
    import distributions.hp.random
    means = [1.0 * i for i in range(-2, 3)]
    variances = [10.0 ** i for i in range(-3, 4)]
    for mean, variance in itertools.product(means, variances):
        # Assume scipy.stats is correct
        # yield _test_normal_draw, scipy_normal_draw, mean, variance
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
    require_cython()
    import distributions.hp.random
    nus = [1.5 ** i for i in range(-10, 11)]
    for nu in nus:
        # Assume scipy.stats is correct
        # yield _test_chisq_draw, scipy.stats.chi2.rvs, nu
        _test_chisq_draw(distributions.hp.random.sample_chisq, nu)


def _test_chisq_draw(draw, nu):
    samples = [draw(nu) for _ in range(SAMPLES)]
    assert_normal(numpy.mean(samples), nu, numpy.sqrt(2 * nu / SAMPLES))


def test_sample_pair_from_urn():
    require_cython()
    import distributions.lp.random
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
    require_cython()
    import distributions.lp.random
    for size in range(1, 100):
        scores = numpy.random.normal(size=size).tolist()
        for _ in xrange(size):
            sample, prob1 = distributions.lp.random.sample_prob_from_scores(
                scores)
            assert 0 <= sample and sample < size
            prob2 = distributions.lp.random.prob_from_scores(
                sample,
                scores)
            assert_close(
                prob1,
                prob2,
                err_msg='sample_prob_from_scores != prob_from_scores')


def test_sample_prob_from_scores():
    require_cython()
    import distributions.lp.random
    for size in range(1, 10):
        scores = numpy.random.normal(size=size).tolist()

        def sampler():
            return distributions.lp.random.sample_prob_from_scores(scores)

        assert_samples_match_scores(sampler)


def test_log_sum_exp():
    require_cython()
    import distributions.lp.random

    for size in xrange(20):
        scores = numpy.random.normal(size=size).tolist()
        expected = numpy.logaddexp.reduce(scores) if size else 0.0
        actual = distributions.lp.random.log_sum_exp(scores)
        assert_close(actual, expected, err_msg='log_sum_exp')


def test_sample_discrete():
    require_cython()
    import distributions.lp.random

    assert_equal(
        distributions.lp.random.sample_discrete(
            numpy.array([.5], dtype=numpy.float32)),
        0)
    assert_equal(
        distributions.lp.random.sample_discrete(
            numpy.array([1.], dtype=numpy.float32)),
        0)
    assert_equal(
        distributions.lp.random.sample_discrete(
            numpy.array([1e-3], dtype=numpy.float32)),
        0)
    assert_equal(
        distributions.lp.random.sample_discrete(
            numpy.array([1 - 1e-3, 1e-3], dtype=numpy.float32)),
        0)
    assert_equal(
        distributions.lp.random.sample_discrete(
            numpy.array([1e-3, 1 - 1e-3], dtype=numpy.float32)),
        1)


def random_orthogonal_matrix(m, n):
    A, _ = numpy.linalg.qr(numpy.random.random((m, n)))
    return A


def random_orthonormal_matrix(n):
    A = random_orthogonal_matrix(n, n)
    return A


def test_sample_iw():

    Q = random_orthonormal_matrix(2)
    nu = 4
    S = numpy.dot(Q, numpy.dot(numpy.diag([1.0, 0.5]), Q.T))

    true_mean = 1. / (nu - S.shape[0] - 1) * S

    ntries = 100
    samples = []
    while ntries:
        samples.extend([sample_inverse_wishart(nu, S) for _ in xrange(10000)])
        mean = sum(samples) / len(samples)
        diff = numpy.linalg.norm(true_mean - mean)
        if diff <= 0.1:
            return
        ntries -= 1

    assert_true(False, "mean did not converge")


def test_score_student_t_scalar_equiv():
    values = (
        (1.2, 5., -0.2, 0.7),
        (-3., 3., 1.2, 1.3),
    )
    for x, nu, mu, sigmasq in values:
        mv_args = [
            numpy.array([x]),
            nu,
            numpy.array([mu]),
            numpy.array([[sigmasq]])]

        scalar_score = scalar_score_student_t(x, nu, mu, sigmasq)
        dbg_mv_score = dbg_score_student_t(*mv_args)
        lp_mv_score = lp_score_student_t(*mv_args)

        assert_close(scalar_score, dbg_mv_score)
        assert_close(scalar_score, lp_mv_score)
        assert_close(dbg_mv_score, lp_mv_score)


def test_score_student_t_dbg_lp_equiv():
    seed_all(0)

    def random_vec(dim):
        return numpy.random.uniform(low=-3., high=3., size=dim)

    def random_cov(dim):
        Q = random_orthonormal_matrix(dim)
        return numpy.dot(Q, Q.T)

    def random_values(dim):
        return (random_vec(dim),
                float(dim) + 1.,
                random_vec(dim),
                random_cov(dim))

    values = (
        [random_values(2) for _ in xrange(10)] +
        [random_values(3) for _ in xrange(10)]
    )

    for x, nu, mu, cov in values:
        dbg_mv_score = dbg_score_student_t(x, nu, mu, cov)
        lp_mv_score = lp_score_student_t(x, nu, mu, cov)
        assert_close(dbg_mv_score, lp_mv_score)
