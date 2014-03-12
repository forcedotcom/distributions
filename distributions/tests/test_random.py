import itertools
import numpy
import scipy
from nose.tools import (
    assert_less,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)
from distributions.tests.util import assert_close, assert_samples_match_scores
import distributions.hp.random
import distributions.lp.random
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
    nus = [1.5 ** i for i in range(-10, 11)]
    for nu in nus:
        # Assume scipy.stats is correct
        #yield _test_chisq_draw, scipy.stats.chi2.rvs, nu
        _test_chisq_draw(distributions.hp.random.sample_chisq, nu)


def _test_chisq_draw(draw, nu):
    samples = [draw(nu) for _ in range(SAMPLES)]
    assert_normal(numpy.mean(samples), nu, numpy.sqrt(2 * nu / SAMPLES))


def test_sample_pair_from_urn():
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
    rng = distributions.lp.random.RNG(0)
    for size in range(1, 10):
        scores = numpy.random.normal(size=size).tolist()

        def sampler():
            return distributions.lp.random.sample_prob_from_scores(rng, scores)

        assert_samples_match_scores(sampler)
