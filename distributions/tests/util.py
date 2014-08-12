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

import os
import glob
from collections import defaultdict
from itertools import izip
import math
import numpy
from numpy.testing import assert_array_almost_equal
from nose import SkipTest
from nose.tools import assert_true, assert_less, assert_equal
import importlib
import distributions
from distributions.util import multinomial_goodness_of_fit

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
TOL = 1e-3


def require_cython():
    if not distributions.has_cython:
        raise SkipTest('no cython support')


def seed_all(s):
    import distributions.dbg.random
    distributions.dbg.random.seed(s)
    try:
        import distributions.hp.random
        distributions.hp.random.seed(s)
    except ImportError:
        pass


def list_models():
    result = set()
    for path in glob.glob(os.path.join(ROOT, '*', 'models', '*.p*')):
        dirname, filename = os.path.split(path)
        flavor = os.path.split(os.path.dirname(dirname))[-1]
        name = os.path.splitext(filename)[0]
        if not name.startswith('__'):
            result.add((name, flavor))
    for name, flavor in sorted(result):
        spec = {'flavor': flavor, 'name': name}
        if name.startswith('_'):
            continue
        try:
            import_model(spec)
            yield spec
        except ImportError:
            module_name = 'distributions.{flavor}.models.{name}'.format(**spec)
            print 'failed to import {}'.format(module_name)
            import traceback
            print traceback.format_exc()


def import_model(spec):
    module_name = 'distributions.{flavor}.models.{name}'.format(**spec)
    return importlib.import_module(module_name)


def assert_hasattr(thing, attr):
    assert_true(
        hasattr(thing, attr),
        "{} is missing attribute '{}'".format(thing.__name__, attr))


def print_short(x, size=64):
    string = str(x)
    if len(string) > size:
        string = string[:size - 3] + '...'
    return string


def assert_close(lhs, rhs, tol=TOL, err_msg=None):
    try:
        if isinstance(lhs, dict):
            assert_true(
                isinstance(rhs, dict),
                'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
            assert_equal(set(lhs.keys()), set(rhs.keys()))
            for key, val in lhs.iteritems():
                msg = '{}[{}]'.format(err_msg or '', key)
                assert_close(val, rhs[key], tol, msg)
        elif isinstance(lhs, float) or isinstance(lhs, numpy.float64):
            assert_true(
                isinstance(rhs, float) or isinstance(rhs, numpy.float64),
                'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
            diff = abs(lhs - rhs)
            norm = 1 + abs(lhs) + abs(rhs)
            msg = '{} off by {}% = {}'.format(
                err_msg or '',
                100 * diff / norm,
                diff)
            assert_less(diff, tol * norm, msg)
        elif isinstance(lhs, numpy.ndarray) or isinstance(rhs, numpy.ndarray):
            assert_true(
                (isinstance(lhs, numpy.ndarray) or isinstance(lhs, list)) and
                (isinstance(rhs, numpy.ndarray) or isinstance(rhs, list)),
                'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
            decimal = int(round(-math.log10(tol)))
            assert_array_almost_equal(
                lhs,
                rhs,
                decimal=decimal,
                err_msg=(err_msg or ''))
        elif isinstance(lhs, list) or isinstance(lhs, tuple):
            assert_true(
                isinstance(rhs, list) or isinstance(rhs, tuple),
                'type mismatch: {} vs {}'.format(type(lhs), type(rhs)))
            for pos, (x, y) in enumerate(izip(lhs, rhs)):
                msg = '{}[{}]'.format(err_msg or '', pos)
                assert_close(x, y, tol, msg)
        else:
            assert_equal(lhs, rhs, err_msg)
    except Exception:
        print err_msg or ''
        print 'actual = {}'.format(print_short(lhs))
        print 'expected = {}'.format(print_short(rhs))
        raise


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
