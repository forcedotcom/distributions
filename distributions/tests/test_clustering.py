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

import math
import functools
from collections import defaultdict
import numpy
import numpy.random
from nose.tools import (
    assert_true,
    assert_equal,
    assert_less,
    assert_greater,
    assert_is_instance,
)
from distributions.util import discrete_goodness_of_fit
from distributions.tests.util import (
    require_cython,
    seed_all,
    assert_hasattr,
    assert_close,
)
from distributions.dbg.random import scores_to_probs
require_cython()
from distributions.lp.clustering import (
    count_assignments,
    PitmanYor,
    LowEntropy,
)

MODELS = {
    'PitmanYor': PitmanYor,
    'LowEntropy': LowEntropy,
}

SAMPLE_COUNT = 2000
MIN_GOODNESS_OF_FIT = 1e-3


def iter_examples(Model):
    assert_hasattr(Model, 'EXAMPLES')
    EXAMPLES = Model.EXAMPLES
    assert_is_instance(EXAMPLES, list)
    assert_true(EXAMPLES, 'no examples provided')
    for i, EXAMPLE in enumerate(EXAMPLES):
        print 'example {}/{}'.format(1 + i, len(Model.EXAMPLES))
        yield EXAMPLE


def for_each_model(*filters):
    '''
    Run one test per Model, filtering out inappropriate Models for test.
    '''
    def filtered(test_fun):

        @functools.wraps(test_fun)
        def test_one_model(name):
            Model = MODELS[name]
            for EXAMPLE in iter_examples(Model):
                seed_all(0)
                test_fun(Model, EXAMPLE)

        @functools.wraps(test_fun)
        def test_all_models():
            for name, Model in MODELS.iteritems():
                if all(f(Model) for f in filters):
                    yield test_one_model, name

        return test_all_models
    return filtered


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


@for_each_model()
def test_load_and_dump(Model, EXAMPLE):
    model = Model()
    model.load(EXAMPLE)
    expected = EXAMPLE
    actual = model.dump()
    assert_close(expected, actual)


def iter_valid_sizes(example, max_size, min_size=2):
    max_size = 5
    dataset_size = example.get('dataset_size', float('inf'))
    sizes = [
        size
        for size in xrange(min_size, max_size + 1)
        if size <= dataset_size
    ]
    assert sizes, 'no valid sizes to test'
    for size in sizes:
        print 'size = {}'.format(size)
        yield size


@for_each_model()
def test_sample_matches_score_counts(Model, EXAMPLE):
    for size in iter_valid_sizes(EXAMPLE, max_size=6):
        model = Model()
        model.load(EXAMPLE)

        samples = []
        probs_dict = {}
        for _ in xrange(SAMPLE_COUNT):
            value = model.sample_assignments(size)
            sample = canonicalize(value)
            samples.append(sample)
            if sample not in probs_dict:
                assignments = dict(enumerate(value))
                counts = count_assignments(assignments)
                prob = math.exp(model.score_counts(counts))
                probs_dict[sample] = prob

        # renormalize here; test normalization separately
        total = sum(probs_dict.values())
        for key in probs_dict:
            probs_dict[key] /= total

        gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        print '{} gof = {:0.3g}'.format(Model.__name__, gof)
        assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_model()
def test_score_counts_is_normalized(Model, EXAMPLE):
    if Model.__name__ == 'LowEntropy':
        print 'WARNING LowEntropy.score has low-precision normalization'
        tol = 1e0
    else:
        tol = 1e-2

    for sample_size in iter_valid_sizes(EXAMPLE, max_size=10):
        model = Model()
        model.load(EXAMPLE)

        probs_dict = {}
        for _ in xrange(SAMPLE_COUNT):
            value = model.sample_assignments(sample_size)
            sample = canonicalize(value)
            if sample not in probs_dict:
                assignments = dict(enumerate(value))
                counts = count_assignments(assignments)
                prob = math.exp(model.score_counts(counts))
                probs_dict[sample] = prob

        total = sum(probs_dict.values())
        assert_less(abs(total - 1), tol, 'not normalized: {}'.format(total))


def add_to_counts(counts, pos):
    counts = counts[:]
    counts[pos] += 1
    return counts


@for_each_model()
def test_score_add_value_matches_score_counts(Model, EXAMPLE):
    for larger_size in iter_valid_sizes(EXAMPLE, min_size=2, max_size=6):
        sample_size = larger_size - 1
        model = Model()
        model.load(EXAMPLE)

        samples = set(
            canonicalize(model.sample_assignments(sample_size))
            for _ in xrange(SAMPLE_COUNT)
        )

        for sample in samples:
            nonempty_group_count = len(sample)
            counts = map(len, sample)
            actual = numpy.zeros(len(counts) + 1)
            expected = numpy.zeros(len(counts) + 1)

            # add to existing group
            for i, group in enumerate(sample):
                group_size = len(sample[i])
                expected[i] = model.score_counts(add_to_counts(counts, i))
                actual[i] = model.score_add_value(
                    group_size,
                    nonempty_group_count,
                    sample_size)

            # add to new group
            i = len(counts)
            group_size = 0
            expected[i] = model.score_counts(counts + [1])
            actual[i] = model.score_add_value(
                group_size,
                nonempty_group_count,
                sample_size)

            actual = scores_to_probs(actual)
            expected = scores_to_probs(expected)
            print actual, expected
            assert_close(actual, expected, tol=0.05)


@for_each_model(lambda Model: hasattr(Model, 'Mixture'))
def test_mixture_score_matches_score_add_value(Model, EXAMPLE):
    model = Model()
    model.load(EXAMPLE)

    value = model.sample_assignments(SAMPLE_COUNT)
    assignments = dict(enumerate(value))
    nonempty_counts = count_assignments(assignments)
    nonempty_group_count = len(nonempty_counts)
    assert_greater(nonempty_group_count, 1, "test is inaccurate")

    for empty_group_count in [1, 10]:
        print 'empty_group_count =', empty_group_count
        counts = nonempty_counts + [0] * empty_group_count
        numpy.random.shuffle(counts)

        expected = [
            model.score_add_value(
                group_size,
                nonempty_group_count,
                SAMPLE_COUNT,
                empty_group_count)
            for group_size in counts
        ]

        mixture = Model.Mixture()
        for count in counts:
            mixture.append(count)
        mixture.init(model)
        noise = numpy.random.randn(len(counts))
        actual = numpy.zeros(len(counts), dtype=numpy.float32)
        actual[:] = noise
        mixture.score(model, actual)

        print 'counts =', counts
        assert_close(actual, expected)

        empty_groupids = frozenset(mixture.empty_groupids)
        assert_equal(len(empty_groupids), empty_group_count)
        for groupid in empty_groupids:
            assert_equal(counts[groupid], 0)
