import itertools
import numpy
import scipy
from nose.tools import assert_less
from distributions import cRandom


SAMPLES = 1000


def assert_close(x, y, sigma, stddevs=4.0):
    '''
    Assert that the difference between two values is within a few standard
    deviations of the predicted [normally distributed] error of zero.
    '''
    assert_less(x, y + sigma * stddevs)
    assert_less(y, x + sigma * stddevs)


def scipy_normal_draw(mean, variance):
    return scipy.stats.norm.rvs(mean, numpy.sqrt(variance))


def test_normal_draw():
    means = [1.0 * i for i in range(-2, 3)]
    variances = [10.0 ** i for i in range(-3, 4)]
    for mean, variance in itertools.product(means, variances):
        # Assume scipy.stats is correct
        #yield _test_normal_draw, scipy_normal_draw, mean, variance
        _test_normal_draw(cRandom.sample_normal, mean, variance)


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
        _test_chisq_draw(cRandom.sample_chisq, nu)


def _test_chisq_draw(draw, nu):
    samples = [draw(nu) for _ in range(SAMPLES)]
    assert_close(numpy.mean(samples), nu, numpy.sqrt(2 * nu / SAMPLES))
