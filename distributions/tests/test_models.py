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

    model = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']
    for value in values:
        assert_is_instance(value, module.Value)

    group1 = module.Group()
    group1.init(model)
    for value in values:
        group1.add_value(model, value)
    group2 = module.Group.from_values(model, values)
    assert_close(group1.dump(), group2.dump())

    group = module.Group.from_values(model, values)
    dumped = group.dump()
    group.init(model)
    group.load(dumped)
    assert_close(group.dump(), dumped)

    for value in values:
        group2.remove_value(model, value)
    assert_not_equal(group1, group2)
    group2.merge(model, group1)

    for value in values:
        module.score_value(model, group1, value)
    for _ in xrange(10):
        value = module.sample_value(model, group1)
        module.score_value(model, group1, value)
        module.sample_group(model, 10)
    module.score_group(model, group1)
    module.score_group(model, group2)

    assert_close(model.dump(), EXAMPLE['shared'])


@for_each_model(lambda module: hasattr(module.Shared, 'load_protobuf'))
def test_protbuf(module, EXAMPLE):
    if not has_protobuf:
        raise SkipTest('protobuf not available')
    model = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']
    group = module.Group.from_values(model, values)
    Message = getattr(distributions.io.schema_pb2, module.NAME)

    message = Message()
    model.dump_protobuf(message)
    model2 = module.Shared()
    model2.load_protobuf(message)
    assert_close(model2.dump(), model.dump())

    message.Clear()
    dumped = model.dump()
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

    model = module.Shared.from_dict(EXAMPLE['shared'])
    #model.realize()
    #values = model['values'][:]

    values = []
    group = module.Group.from_values(model)
    score = 0.0
    assert_close(
        module.score_group(model, group), score, err_msg='p(empty) != 1')

    for _ in range(DATA_COUNT):
        value = module.sample_value(model, group)
        values.append(value)
        score += module.score_value(model, group, value)
        group.add_value(model, value)

    group_all = module.Group.from_dict(group.dump())
    assert_close(
        score,
        module.score_group(model, group),
        err_msg='p(x1,...,xn) != p(x1) p(x2|x1) p(xn|...)')

    numpy.random.shuffle(values)

    for value in values:
        group.remove_value(model, value)

    group_empty = module.Group.from_values(model)
    assert_close(
        group.dump(),
        group_empty.dump(),
        err_msg='group + values - values != group')

    numpy.random.shuffle(values)
    for value in values:
        group.add_value(model, value)
    assert_close(
        group.dump(),
        group_all.dump(),
        err_msg='group - values + values != group')


@for_each_model()
def test_add_merge(module, EXAMPLE):
    # Test group_add_value, group_merge
    model = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values'][:]
    numpy.random.shuffle(values)
    group = module.Group.from_values(model, values)

    for i in xrange(len(values) + 1):
        numpy.random.shuffle(values)
        group1 = module.Group.from_values(model, values[:i])
        group2 = module.Group.from_values(model, values[i:])
        group1.merge(model, group2)
        assert_close(group.dump(), group1.dump())


@for_each_model()
def test_group_merge(module, EXAMPLE):
    model = module.Shared.from_dict(EXAMPLE['shared'])
    group1 = module.Group.from_values(model)
    group2 = module.Group.from_values(model)
    expected = module.Group.from_values(model)
    actual = module.Group.from_values(model)
    for _ in xrange(100):
        value = module.sample_value(model, expected)
        expected.add_value(model, value)
        group1.add_value(model, value)

        value = module.sample_value(model, expected)
        expected.add_value(model, value)
        group2.add_value(model, value)

        actual.load(group1.dump())
        actual.merge(model, group2)
        assert_close(actual.dump(), expected.dump())


@for_each_model()
def test_sample_seed(module, EXAMPLE):
    model = module.Shared.from_dict(EXAMPLE['shared'])

    seed_all(0)
    group1 = module.Group.from_values(model)
    values1 = [module.sample_value(model, group1) for _ in xrange(DATA_COUNT)]

    seed_all(0)
    group2 = module.Group.from_values(model)
    values2 = [module.sample_value(model, group2) for _ in xrange(DATA_COUNT)]

    assert_close(values1, values2, err_msg='values')


@for_each_model()
def test_sample_value(module, EXAMPLE):
    seed_all(0)
    model = module.Shared.from_dict(EXAMPLE['shared'])
    for values in [[], EXAMPLE['values']]:
        group = module.Group.from_values(model, values)
        samples = [
            module.sample_value(model, group) for _ in xrange(SAMPLE_COUNT)]
        if module.Value == int:
            probs_dict = {
                value: math.exp(module.score_value(model, group, value))
                for value in set(samples)
            }
            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        elif module.Value == float:
            probs = numpy.exp([
                module.score_value(model, group, value)
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
    model = module.Shared.from_dict(EXAMPLE['shared'])
    for values in [[], EXAMPLE['values']]:
        if module.Value == int:
            samples = []
            probs_dict = {}
            for _ in xrange(SAMPLE_COUNT):
                values = module.sample_group(model, SIZE)
                sample = tuple(values)
                samples.append(sample)
                group = module.Group.from_values(model, values)
                probs_dict[sample] = math.exp(module.score_group(model, group))
            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        else:
            raise SkipTest('Not implemented for {}'.format(module.Value))
        print '{} gof = {:0.3g}'.format(module.__name__, gof)
        assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_model(lambda module: hasattr(module.Shared, 'scorer_create'))
def test_scorer(module, EXAMPLE):
    model = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']

    group = module.Group.from_values(model)
    scorer1 = model.scorer_create()
    scorer2 = model.scorer_create(group)
    for value in values:
        score1 = model.scorer_eval(scorer1, value)
        score2 = model.scorer_eval(scorer2, value)
        score3 = module.score_value(model, group, value)
        assert_all_close([score1, score2, score3])


@for_each_model(lambda module: hasattr(module, 'Mixture'))
def test_mixture_runs(module, EXAMPLE):
    model = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']

    mixture = module.Mixture()
    for value in values:
        mixture.append(module.Group.from_values(model, [value]))
    mixture.init(model)

    groupids = []
    for value in values:
        scores = numpy.zeros(len(mixture), dtype=numpy.float32)
        mixture.score_value(model, value, scores)
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        mixture.add_value(model, groupid, value)
        groupids.append(groupid)

    mixture.add_group(model)
    assert len(mixture) == len(values) + 1
    scores = numpy.zeros(len(mixture), dtype=numpy.float32)

    for value, groupid in zip(values, groupids):
        mixture.remove_value(model, groupid, value)

    mixture.remove_group(model, 0)
    mixture.remove_group(model, len(mixture) - 1)
    assert len(mixture) == len(values) - 1

    for value in values:
        scores = numpy.zeros(len(mixture), dtype=numpy.float32)
        mixture.score_value(model, value, scores)
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        mixture.add_value(model, groupid, value)


@for_each_model(lambda module: hasattr(module, 'Mixture'))
def test_mixture_score(module, EXAMPLE):
    model = module.Shared.from_dict(EXAMPLE['shared'])
    values = EXAMPLE['values']

    groups = [module.Group.from_values(model, [value]) for value in values]
    mixture = module.Mixture()
    for group in groups:
        mixture.append(group)
    mixture.init(model)

    def check_scores():
        expected = [
            module.score_value(model, group, value) for group in groups]
        actual = numpy.zeros(len(mixture), dtype=numpy.float32)
        noise = numpy.random.randn(len(actual))
        actual += noise
        mixture.score_value(model, value, actual)
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
        groups[groupid].add_value(model, value)
        mixture.add_value(model, groupid, value)
        groupids.append(groupid)

    print 'removing'
    for value, groupid in zip(values, groupids):
        groups[groupid].remove_value(model, value)
        mixture.remove_value(model, groupid, value)
        scores = check_scores()
