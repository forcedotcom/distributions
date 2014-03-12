import math
from collections import defaultdict
from nose import SkipTest
from nose.tools import assert_less, assert_greater
from distributions.util import discrete_goodness_of_fit
from distributions.tests.util import seed_all
from distributions.lp.clustering import (
    count_assignments,
    PitmanYor,
    LowEntropy,
)


MODELS = [
    PitmanYor,
    LowEntropy,
]

SAMPLE_COUNT = 10000
MAX_SIZE = 5
MIN_GOODNESS_OF_FIT = 1e-3


def canonicalize(assignments):
    groups = defaultdict(lambda: [])
    for value, group in enumerate(assignments):
        groups[group].append(value)
    result = []
    for group in groups.itervalues():
        group.sort()
        result.append(tuple(group))
    result.sort()
    return tuple(result)


def test_models(Model=None, size=None):
    seed_all(0)
    for Model in MODELS:
        for size in xrange(2, MAX_SIZE + 1):
            yield _test_models, Model, size


def _test_models(Model, size):
        model = Model()

        if Model.__name__ == 'LowEntropy':
            raise SkipTest('FIXME LowEntropy.score_counts is not normalized')

        for i, EXAMPLE in enumerate(Model.EXAMPLES):
            print 'Example {}'.format(i)
            model.load(EXAMPLE)
            samples = []
            probs_dict = {}
            for _ in xrange(SAMPLE_COUNT):
                value = model.sample_assignments(size)
                assignments = dict(enumerate(value))
                counts = count_assignments(assignments)
                prob = math.exp(model.score_counts(counts))
                sample = canonicalize(value)
                samples.append(sample)
                probs_dict[sample] = prob

            total = sum(probs_dict.values())
            assert_less(
                abs(total - 1),
                1e-2,
                'not normalized: {}'.format(total))

            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
            print '{} gof = {:0.3g}'.format(Model.__name__, gof)
            assert_greater(gof, MIN_GOODNESS_OF_FIT)
