import os
import glob
from collections import defaultdict
from itertools import izip
import numpy
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_less, assert_equal
import importlib
from distributions.util import multinomial_goodness_of_fit
import distributions.dbg.random
import distributions.hp.random
#import distributions.lp.random

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def seed_all(s):
    distributions.dbg.random.seed(s)
    distributions.hp.random.seed(s)
    #distributions.lp.random.seed(s)


def list_models():
    result = set()
    for path in glob.glob(os.path.join(ROOT, '*', 'models', '*.p*')):
        dirname, filename = os.path.split(path)
        flavor = os.path.split(os.path.dirname(dirname))[-1]
        name = os.path.splitext(filename)[0]
        if not name.startswith('__'):
            result.add((name, flavor))
    for name, flavor in sorted(result):
        yield {'flavor': flavor, 'name': name}


def import_model(spec):
    module_name = 'distributions.{flavor}.models.{name}'.format(**spec)
    return importlib.import_module(module_name)


def assert_hasattr(thing, attr):
    assert_true(
        hasattr(thing, attr),
        "{} is missing attribute '{}'".format(thing.__name__, attr))


def assert_close(lhs, rhs, percent=0.1, tol=1e-3, err_msg=None):
    if isinstance(lhs, dict):
        assert_true(
            isinstance(rhs, dict),
            'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
        assert_equal(set(lhs.keys()), set(rhs.keys()))
        for key, val in lhs.iteritems():
            msg = '{}[{}]'.format(err_msg or '', key)
            assert_close(val, rhs[key], percent, tol, msg)
    elif isinstance(lhs, float) or isinstance(lhs, numpy.float64):
        assert_true(
            isinstance(rhs, float) or isinstance(rhs, numpy.float64),
            'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
        diff = abs(lhs - rhs)
        norm = (abs(lhs) + abs(rhs)) * (percent / 100) + tol
        msg = '{} off by {}% = {}'.format(err_msg, 100 * diff / norm, diff)
        assert_less(diff, norm, msg)
    elif isinstance(lhs, numpy.ndarray) or isinstance(lhs, list):
        assert_true(
            isinstance(rhs, numpy.ndarray) or isinstance(rhs, list),
            'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
        assert_array_almost_equal(lhs, rhs, err_msg=(err_msg or ''))
    else:
        assert_equal(lhs, rhs, err_msg)


def assert_all_close(collection, **kwargs):
    for i1, item1 in enumerate(collection[:-1]):
        for item2 in collection[i1 + 1:]:
            assert_close(item1, item2, **kwargs)


def collect_samples_and_scores(sampler, total_count=10000):
    '''
    Collect samples and MC estimates of sample probabilities.

    Inputs:
        - sampler generates (sample, prob) pairs.  samples must be hashable.
          probs may be randomized, but must be unbiased and low-variance.
        - total_count samples are drawn in total.
        - tol is the minimum goodness of fit allowed to pass the test.
    Returns:
        - counts : key -> int
        - probs : key -> float
    '''
    counts = defaultdict(lambda: 0)
    probs = defaultdict(lambda: 0.0)
    for _ in xrange(total_count):
        sample, prob = sampler()
        counts[sample] += 1
        probs[sample] += prob

    for key, count in counts.iteritems():
        probs[key] /= count
    total_prob = sum(probs.itervalues())
    assert_close(total_prob, 1.0, tol=1e-2, err_msg='total_prob is biased')

    return counts, probs


def assert_counts_match_probs(counts, probs, tol=1e-3):
    '''
    Check goodness of fit of observed counts to predicted probabilities
    using Pearson's chi-squared test.

    Inputs:
        - counts : key -> int
        - probs : key -> float
    '''
    keys = counts.keys()
    probs = [probs[key] for key in keys]
    counts = [counts[key] for key in keys]
    total_count = sum(counts)

    print 'EXPECT\tACTUAL\tVALUE'
    for prob, count, key in sorted(izip(probs, counts, keys), reverse=True):
        expect = prob * total_count
        print '{:0.1f}\t{}\t{}'.format(expect, count, key)

    gof = multinomial_goodness_of_fit(probs, counts, total_count)
    print 'goodness of fit = {}'.format(gof)
    assert gof > tol, 'failed with goodness of fit {}'.format(gof)


def assert_samples_match_scores(sampler, total_count=10000, tol=1e-3):
    '''
    Test that a discrete sampler is distributed according to its scores.

    Inputs:
        - sampler generates (sample, prob) pairs.  samples must be hashable.
          probs may be randomized, but must be unbiased and low-variance.
        - total_count samples are drawn in total.
        - tol is the minimum goodness of fit allowed to pass the test.
    '''
    counts, probs = collect_samples_and_scores(sampler, total_count)
    assert_counts_match_probs(counts, probs)
