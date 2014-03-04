import itertools
import numpy
import scipy
from nose.tools import (
    assert_less,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)
import distributions.cRandom
from distributions.random import sample_stick, sample_discrete_log


SAMPLES = 1000


def assert_close(x, y, sigma, stddevs=4.0):
    '''
    Assert that the difference between two values is within a few standard
    deviations of the predicted [normally distributed] error of zero.
    '''
    assert_less(x, y + sigma * stddevs)
    assert_less(y, x + sigma * stddevs)


def test_seed():
    global_rng = distributions.cRandom.random
    distributions.cRandom.seed(0)
    values1 = [global_rng() for _ in xrange(10)]
    distributions.cRandom.seed(0)
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
        _test_normal_draw(distributions.cRandom.sample_normal, mean, variance)


def _test_normal_draw(draw, mean, variance):
    samples = [draw(mean, variance) for _ in range(SAMPLES)]
    assert_close(numpy.mean(samples), mean, numpy.sqrt(variance / SAMPLES))
    error = numpy.array(samples) - mean
    chisq = numpy.dot(error, error) / variance
    assert_close(chisq, SAMPLES, 2 * SAMPLES)


def test_chisq_draw():
    nus = [1.5 ** i for i in range(-10, 11)]
    for nu in nus:
        # Assume scipy.stats is correct
        #yield _test_chisq_draw, scipy.stats.chi2.rvs, nu
        _test_chisq_draw(distributions.cRandom.sample_chisq, nu)


def _test_chisq_draw(draw, nu):
    samples = [draw(nu) for _ in range(SAMPLES)]
    assert_close(numpy.mean(samples), nu, numpy.sqrt(2 * nu / SAMPLES))
