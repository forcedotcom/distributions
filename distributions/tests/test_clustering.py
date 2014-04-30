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
from nose import SkipTest
from nose.tools import (
    assert_true,
    assert_equal,
    assert_less,
    assert_greater,
    assert_is_instance,
)
from distributions.dbg.random import sample_discrete
from distributions.util import discrete_goodness_of_fit
from distributions.tests.util import (
    require_cython,
    seed_all,
    assert_hasattr,
    assert_close,
)
from distributions.dbg.random import scores_to_probs
import distributions.dbg.clustering
require_cython()
import distributions.lp.clustering
from distributions.lp.clustering import count_assignments
from distributions.lp.mixture import MixtureIdTracker

MODELS = {
    'dbg.LowEntropy': distributions.dbg.clustering.LowEntropy,
    'lp.PitmanYor': distributions.lp.clustering.PitmanYor,
    'lp.LowEntropy': distributions.lp.clustering.LowEntropy,
}

SKIP_EXPENSIVE_TESTS = False
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
                if SKIP_EXPENSIVE_TESTS and name.startswith('dbg'):
                    sample_count = SAMPLE_COUNT / 10
                else:
                    sample_count = SAMPLE_COUNT
                test_fun(Model, EXAMPLE, sample_count)

        @functools.wraps(test_fun)
        def test_all_models():
            for name, Model in sorted(MODELS.iteritems()):
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
def test_load_and_dump(Model, EXAMPLE, *unused):
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
        print 'sample_size = {}'.format(size)
        yield size


@for_each_model()
def test_sample_matches_score_counts(Model, EXAMPLE, sample_count):
    for size in iter_valid_sizes(EXAMPLE, max_size=10):
        model = Model()
        model.load(EXAMPLE)

        samples = []
        probs_dict = {}
        for _ in xrange(sample_count):
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
def test_score_counts_is_normalized(Model, EXAMPLE, sample_count):

    for sample_size in iter_valid_sizes(EXAMPLE, max_size=10):
        model = Model()
        model.load(EXAMPLE)

        if Model.__name__ == 'LowEntropy' and sample_size < model.dataset_size:
            print 'WARNING LowEntropy.score_counts normalization is imprecise'
            print '  when sample_size < dataset_size'
            tol = 0.5
        else:
            tol = 0.01

        probs_dict = {}
        for _ in xrange(sample_count):
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
def test_score_add_value_matches_score_counts(Model, EXAMPLE, sample_count):
    for sample_size in iter_valid_sizes(EXAMPLE, min_size=2, max_size=10):
        model = Model()
        model.load(EXAMPLE)

        samples = set(
            canonicalize(model.sample_assignments(sample_size - 1))
            for _ in xrange(sample_count)
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
                    sample_size - 1)

            # add to new group
            i = len(counts)
            group_size = 0
            expected[i] = model.score_counts(counts + [1])
            actual[i] = model.score_add_value(
                group_size,
                nonempty_group_count,
                sample_size - 1)

            actual = scores_to_probs(actual)
            expected = scores_to_probs(expected)
            print actual, expected
            assert_close(actual, expected, tol=0.05)


@for_each_model(lambda Model: hasattr(Model, 'Mixture'))
def test_mixture_score_matches_score_add_value(Model, EXAMPLE, *unused):
    sample_count = 200
    model = Model()
    model.load(EXAMPLE)

    if Model.__name__ == 'LowEntropy' and sample_count > model.dataset_size:
        raise SkipTest('skipping trivial example')

    assignment_vector = model.sample_assignments(sample_count)
    assignments = dict(enumerate(assignment_vector))
    nonempty_counts = count_assignments(assignments)
    nonempty_group_count = len(nonempty_counts)
    assert_greater(nonempty_group_count, 1, "test is inaccurate")

    def check_counts(mixture, counts, empty_group_count):
        #print 'counts =', counts
        empty_groupids = frozenset(mixture.empty_groupids)
        assert_equal(len(empty_groupids), empty_group_count)
        for groupid in empty_groupids:
            assert_equal(counts[groupid], 0)

    def check_scores(mixture, counts, empty_group_count):
        sample_count = sum(counts)
        nonempty_group_count = len(counts) - empty_group_count
        expected = [
            model.score_add_value(
                group_size,
                nonempty_group_count,
                sample_count,
                empty_group_count)
            for group_size in counts
        ]
        noise = numpy.random.randn(len(counts))
        actual = numpy.zeros(len(counts), dtype=numpy.float32)
        actual[:] = noise
        mixture.score_value(model, actual)
        assert_close(actual, expected)
        return actual

    for empty_group_count in [1, 10]:
        print 'empty_group_count =', empty_group_count
        counts = nonempty_counts + [0] * empty_group_count
        numpy.random.shuffle(counts)
        mixture = Model.Mixture()
        id_tracker = MixtureIdTracker()

        print 'init'
        mixture.init(model, counts)
        id_tracker.init(len(counts))
        check_counts(mixture, counts, empty_group_count)
        check_scores(mixture, counts, empty_group_count)

        print 'adding'
        groupids = []
        for _ in xrange(sample_count):
            check_counts(mixture, counts, empty_group_count)
            scores = check_scores(mixture, counts, empty_group_count)
            probs = scores_to_probs(scores)
            groupid = sample_discrete(probs)
            expected_group_added = (counts[groupid] == 0)
            counts[groupid] += 1
            actual_group_added = mixture.add_value(model, groupid)
            assert_equal(actual_group_added, expected_group_added)
            groupids.append(groupid)
            if actual_group_added:
                id_tracker.add_group()
                counts.append(0)

        check_counts(mixture, counts, empty_group_count)
        check_scores(mixture, counts, empty_group_count)

        print 'removing'
        for global_groupid in groupids:
            groupid = id_tracker.global_to_packed(global_groupid)
            counts[groupid] -= 1
            expected_group_removed = (counts[groupid] == 0)
            actual_group_removed = mixture.remove_value(model, groupid)
            assert_equal(actual_group_removed, expected_group_removed)
            if expected_group_removed:
                id_tracker.remove_group(groupid)
                back = counts.pop()
                if groupid < len(counts):
                    counts[groupid] = back
            check_counts(mixture, counts, empty_group_count)
            check_scores(mixture, counts, empty_group_count)
