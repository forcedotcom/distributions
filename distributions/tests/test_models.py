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
import numpy
import numpy.random
import functools
from nose import SkipTest
from nose.tools import (
    assert_true,
    assert_in,
    assert_is_instance,
    assert_not_equal,
    assert_greater,
)
from distributions.dbg.random import sample_discrete
from distributions.util import (
    scores_to_probs,
    density_goodness_of_fit,
    discrete_goodness_of_fit,
)
from distributions.tests.util import (
    assert_hasattr,
    assert_close,
    assert_all_close,
    list_models,
    import_model,
    seed_all,
)

try:
    import distributions.io.schema_pb2
    has_protobuf = True
except ImportError:
    has_protobuf = False

DATA_COUNT = 20
SAMPLE_COUNT = 1000
MIN_GOODNESS_OF_FIT = 1e-3

MODULES = {
    '{flavor}.models.{name}'.format(**spec): import_model(spec)
    for spec in list_models()
}


def iter_examples(module):
    assert_hasattr(module, 'EXAMPLES')
    EXAMPLES = module.EXAMPLES
    assert_is_instance(EXAMPLES, list)
    assert_true(EXAMPLES, 'no examples provided')
    for i, EXAMPLE in enumerate(EXAMPLES):
        print 'example {}/{}'.format(1 + i, len(EXAMPLES))
        assert_in('shared', EXAMPLE)
        assert_in('values', EXAMPLE)
        values = EXAMPLE['values']
        assert_is_instance(values, list)
        count = len(values)
        assert_true(
            count >= 7,
            'Add more example values (expected >= 7, found {})'.format(count))
        yield EXAMPLE


def for_each_model(*filters):
    '''
    Run one test per Model, filtering out inappropriate Models for test.
    '''
    def filtered(test_fun):

        @functools.wraps(test_fun)
        def test_one_model(name):
            module = MODULES[name]
            assert_hasattr(module, 'Shared')
            for EXAMPLE in iter_examples(module):
                test_fun(module, EXAMPLE)

        @functools.wraps(test_fun)
        def test_all_models():
            for name in MODULES:
                module = MODULES[name]
                if all(f(module) for f in filters):
                    yield test_one_model, name

        return test_all_models
    return filtered


@for_each_model()
def test_interface(module, EXAMPLE):
    for typename in ['Value', 'Group']:
        assert_hasattr(module, typename)
        assert_is_instance(getattr(module, typename), type)

    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']
    for value in values:
        assert_is_instance(value, module.Value)

    group1 = module.Group()
    group1.init(shared)
    for value in values:
        group1.add_value(shared, value)
    group2 = module.Group.from_values(shared, values)
    assert_close(group1.dump(), group2.dump())

    group = module.Group.from_values(shared, values)
    dumped = group.dump()
    group.init(shared)
    group.load(dumped)
    assert_close(group.dump(), dumped)

    for value in values:
        group2.remove_value(shared, value)
    assert_not_equal(group1, group2)
    group2.merge(shared, group1)

    for value in values:
        group1.score_value(shared, value)
    for _ in xrange(10):
        value = module.sample_value(shared, group1)
        group1.score_value(shared, value)
        module.sample_group(shared, 10)
    module.score_group(shared, group1)
    module.score_group(shared, group2)

    assert_close(shared.dump(), EXAMPLE['shared'])


@for_each_model(lambda module: hasattr(module.Shared, 'load_protobuf'))
def test_protbuf(module, EXAMPLE):
    if not has_protobuf:
        raise SkipTest('protobuf not available')
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']
    group = module.Group.from_values(shared, values)
    Message = getattr(distributions.io.schema_pb2, module.NAME)

    message = Message()
    shared.dump_protobuf(message)
    shared2 = module.Shared()
    shared2.load_protobuf(message)
    assert_close(shared2.dump(), shared.dump())

    message.Clear()
    dumped = shared.dump()
    module.Shared.to_protobuf(dumped, message)
    assert_close(module.Shared.from_protobuf(message), dumped)

    message = Message.Group()
    group.dump_protobuf(message)
    group2 = module.Group()
    group2.load_protobuf(message)
    assert_close(group2.dump(), group.dump())

    message.Clear()
    dumped = group.dump()
    module.Group.to_protobuf(dumped, message)
    assert_close(module.Group.from_protobuf(message), dumped)


@for_each_model()
def test_add_remove(module, EXAMPLE):
    # Test group_add_value, group_remove_value, score_group, score_value

    shared = module.Shared.from_dict(EXAMPLE['shared'])
    #shared.realize()
    #values = shared['values'][:]

    values = []
    group = module.Group.from_values(shared)
    score = 0.0
    assert_close(
        module.score_group(shared, group), score, err_msg='p(empty) != 1')

    for _ in range(DATA_COUNT):
        value = module.sample_value(shared, group)
        values.append(value)
        score += group.score_value(shared, value)
        group.add_value(shared, value)

    group_all = module.Group.from_dict(group.dump())
    assert_close(
        score,
        module.score_group(shared, group),
        err_msg='p(x1,...,xn) != p(x1) p(x2|x1) p(xn|...)')

    numpy.random.shuffle(values)

    for value in values:
        group.remove_value(shared, value)

    group_empty = module.Group.from_values(shared)
    assert_close(
        group.dump(),
        group_empty.dump(),
        err_msg='group + values - values != group')

    numpy.random.shuffle(values)
    for value in values:
        group.add_value(shared, value)
    assert_close(
        group.dump(),
        group_all.dump(),
        err_msg='group - values + values != group')


@for_each_model()
def test_add_merge(module, EXAMPLE):
    # Test group_add_value, group_merge
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values'][:]
    numpy.random.shuffle(values)
    group = module.Group.from_values(shared, values)

    for i in xrange(len(values) + 1):
        numpy.random.shuffle(values)
        group1 = module.Group.from_values(shared, values[:i])
        group2 = module.Group.from_values(shared, values[i:])
        group1.merge(shared, group2)
        assert_close(group.dump(), group1.dump())


@for_each_model()
def test_group_merge(module, EXAMPLE):
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    group1 = module.Group.from_values(shared)
    group2 = module.Group.from_values(shared)
    expected = module.Group.from_values(shared)
    actual = module.Group.from_values(shared)
    for _ in xrange(100):
        value = module.sample_value(shared, expected)
        expected.add_value(shared, value)
        group1.add_value(shared, value)

        value = module.sample_value(shared, expected)
        expected.add_value(shared, value)
        group2.add_value(shared, value)

        actual.load(group1.dump())
        actual.merge(shared, group2)
        assert_close(actual.dump(), expected.dump())


@for_each_model()
def test_sample_seed(module, EXAMPLE):
    shared = module.Shared.from_dict(EXAMPLE['shared'])

    seed_all(0)
    group1 = module.Group.from_values(shared)
    values1 = [module.sample_value(shared, group1) for _ in xrange(DATA_COUNT)]

    seed_all(0)
    group2 = module.Group.from_values(shared)
    values2 = [module.sample_value(shared, group2) for _ in xrange(DATA_COUNT)]

    assert_close(values1, values2, err_msg='values')


@for_each_model()
def test_sample_value(module, EXAMPLE):
    seed_all(0)
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    for values in [[], EXAMPLE['values']]:
        group = module.Group.from_values(shared, values)
        samples = [
            module.sample_value(shared, group) for _ in xrange(SAMPLE_COUNT)]
        if module.Value == int:
            probs_dict = {
                value: math.exp(group.score_value(shared, value))
                for value in set(samples)
            }
            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        elif module.Value == float:
            probs = numpy.exp([
                group.score_value(shared, value)
                for value in samples
            ])
            gof = density_goodness_of_fit(samples, probs, plot=True)
        else:
            raise SkipTest('Not implemented for {}'.format(module.Value))
        print '{} gof = {:0.3g}'.format(module.__name__, gof)
        assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_model()
def test_sample_group(module, EXAMPLE):
    seed_all(0)
    SIZE = 2
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    for values in [[], EXAMPLE['values']]:
        if module.Value == int:
            samples = []
            probs_dict = {}
            for _ in xrange(SAMPLE_COUNT):
                values = module.sample_group(shared, SIZE)
                sample = tuple(values)
                samples.append(sample)
                group = module.Group.from_values(shared, values)
                probs_dict[sample] = math.exp(
                    module.score_group(shared, group))
            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        else:
            raise SkipTest('Not implemented for {}'.format(module.Value))
        print '{} gof = {:0.3g}'.format(module.__name__, gof)
        assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_model(lambda module: hasattr(module.Shared, 'scorer_create'))
def test_scorer(module, EXAMPLE):
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']

    group = module.Group.from_values(shared)
    scorer1 = shared.scorer_create()
    scorer2 = shared.scorer_create(group)
    for value in values:
        score1 = shared.scorer_eval(scorer1, value)
        score2 = shared.scorer_eval(scorer2, value)
        score3 = group.score_value(shared, value)
        assert_all_close([score1, score2, score3])


@for_each_model(lambda module: hasattr(module, 'Mixture'))
def test_mixture_runs(module, EXAMPLE):
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']

    mixture = module.Mixture()
    for value in values:
        mixture.append(module.Group.from_values(shared, [value]))
    mixture.init(shared)

    groupids = []
    for value in values:
        scores = numpy.zeros(len(mixture), dtype=numpy.float32)
        mixture.score_value(shared, value, scores)
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        mixture.add_value(shared, groupid, value)
        groupids.append(groupid)

    mixture.add_group(shared)
    assert len(mixture) == len(values) + 1
    scores = numpy.zeros(len(mixture), dtype=numpy.float32)

    for value, groupid in zip(values, groupids):
        mixture.remove_value(shared, groupid, value)

    mixture.remove_group(shared, 0)
    mixture.remove_group(shared, len(mixture) - 1)
    assert len(mixture) == len(values) - 1

    for value in values:
        scores = numpy.zeros(len(mixture), dtype=numpy.float32)
        mixture.score_value(shared, value, scores)
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        mixture.add_value(shared, groupid, value)


@for_each_model(lambda module: hasattr(module, 'Mixture'))
def test_mixture_score(module, EXAMPLE):
    shared = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']

    groups = [module.Group.from_values(shared, [value]) for value in values]
    mixture = module.Mixture()
    for group in groups:
        mixture.append(group)
    mixture.init(shared)

    def check_scores():
        expected = [
            group.score_value(shared, value) for group in groups]
        actual = numpy.zeros(len(mixture), dtype=numpy.float32)
        noise = numpy.random.randn(len(actual))
        actual += noise
        mixture.score_value(shared, value, actual)
        actual -= noise
        assert_close(actual, expected, err_msg='scores')
        return actual

    print 'init'
    for value in values:
        check_scores()

    print 'adding'
    groupids = []
    for value in values:
        scores = check_scores()
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        groups[groupid].add_value(shared, value)
        mixture.add_value(shared, groupid, value)
        groupids.append(groupid)

    print 'removing'
    for value, groupid in zip(values, groupids):
        groups[groupid].remove_value(shared, value)
        mixture.remove_value(shared, groupid, value)
        scores = check_scores()
