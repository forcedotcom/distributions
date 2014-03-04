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
