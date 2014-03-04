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


def test_multinmoial_goodness_of_fit():
    thresh = 1e-3
    n = int(1e5)
    ds = [3, 10, 20]
    for d in ds:
        for _ in range(5):
            probs = numpy.random.dirichlet([1] * d)
            counts = numpy.random.multinomial(n, probs)
            p_good = multinomial_goodness_of_fit(probs, counts, n)
            assert_greater(p_good, thresh)

        unif_counts = numpy.random.multinomial(n, [1. / d] * d)
        p_bad = multinomial_goodness_of_fit(probs, unif_counts, n)
        assert_less(p_bad, thresh)


def test_bin_samples():
    samples = range(6)
    numpy.random.shuffle(samples)
    counts, bounds = bin_samples(samples, 2)
    assert_list_equal(list(counts), [3, 3])
    assert_list_equal(list(bounds[0]), [0, 3])
    assert_list_equal(list(bounds[1]), [3, 5])
