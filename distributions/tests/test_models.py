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
import random
import numpy
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

DATA_COUNT = 20
SAMPLE_COUNT = 1000
MIN_GOODNESS_OF_FIT = 1e-3

MODULES = {
    '{flavor}.models.{name}'.format(**spec): import_model(spec)
    for spec in list_models()
}


def iter_examples(Model):
    assert_hasattr(Model, 'EXAMPLES')
    EXAMPLES = Model.EXAMPLES
    assert_is_instance(EXAMPLES, list)
    assert_true(EXAMPLES, 'no examples provided')
    for i, EXAMPLE in enumerate(EXAMPLES):
        print 'example {}/{}'.format(1 + i, len(Model.EXAMPLES))
        assert_in('model', EXAMPLE)
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
            assert_hasattr(module, 'Model')
            Model = module.Model
            for EXAMPLE in iter_examples(Model):
                test_fun(Model, EXAMPLE)

        @functools.wraps(test_fun)
        def test_all_models():
            for name in MODULES:
                Model = MODULES[name].Model
                if all(f(Model) for f in filters):
                    yield test_one_model, name

        return test_all_models
    return filtered


@for_each_model()
def test_interface(Model, EXAMPLE):
    for typename in ['Value', 'Group']:
        assert_hasattr(Model, typename)
        assert_is_instance(getattr(Model, typename), type)

    model = Model.model_load(EXAMPLE['model'])
    values = EXAMPLE['values']
    for value in values:
        assert_is_instance(value, Model.Value)

    group1 = model.Group()
    model.group_init(group1)
    for value in values:
        model.group_add_value(group1, value)
    group2 = model.group_create(values)
    assert_close(group1.dump(), group2.dump())

    group = model.group_create(values)
    dumped = group.dump()
    model.group_init(group)
    group.load(dumped)
    assert_close(group.dump(), dumped)

    for value in values:
        model.group_remove_value(group2, value)
    assert_not_equal(group1, group2)
    model.group_merge(group2, group1)

    for value in values:
        model.score_value(group1, value)
    for _ in xrange(10):
        value = model.sample_value(group1)
        model.score_value(group1, value)
        model.sample_group(10)
    model.score_group(group1)
    model.score_group(group2)

    assert_close(model.dump(), EXAMPLE['model'])
    assert_close(model.dump(), Model.model_dump(model))
    assert_close(group1.dump(), Model.group_dump(group1))


@for_each_model()
def test_add_remove(Model, EXAMPLE):
    # Test group_add_value, group_remove_value, score_group, score_value

    model = Model.model_load(EXAMPLE['model'])
    #model.realize()
    #values = model['values'][:]

    values = []
    group = model.group_create()
    score = 0.0
    assert_close(model.score_group(group), score, err_msg='p(empty) != 1')

    for _ in range(DATA_COUNT):
        value = model.sample_value(group)
        values.append(value)
        score += model.score_value(group, value)
        model.group_add_value(group, value)

    group_all = model.group_load(model.group_dump(group))
    assert_close(
        score,
        model.score_group(group),
        err_msg='p(x1,...,xn) != p(x1) p(x2|x1) p(xn|...)')

    random.shuffle(values)

    for value in values:
        model.group_remove_value(group, value)

    group_empty = model.group_create()
    assert_close(
        group.dump(),
        group_empty.dump(),
        err_msg='group + values - values != group')

    random.shuffle(values)
    for value in values:
        model.group_add_value(group, value)
    assert_close(
        group.dump(),
        group_all.dump(),
        err_msg='group - values + values != group')


@for_each_model()
def test_add_merge(Model, EXAMPLE):
    # Test group_add_value, group_merge
    model = Model.model_load(EXAMPLE['model'])
    values = EXAMPLE['values'][:]
    random.shuffle(values)
    group = model.group_create(values)

    for i in xrange(len(values) + 1):
        random.shuffle(values)
        group1 = model.group_create(values[:i])
        group2 = model.group_create(values[i:])
        model.group_merge(group1, group2)
        assert_close(group.dump(), group1.dump())


@for_each_model()
def test_group_merge(Model, EXAMPLE):
    model = Model.model_load(EXAMPLE['model'])
    group1 = model.group_create()
    group2 = model.group_create()
    expected = model.group_create()
    actual = model.group_create()
    for _ in xrange(100):
        value = model.sample_value(expected)
        model.group_add_value(expected, value)
        model.group_add_value(group1, value)

        value = model.sample_value(expected)
        model.group_add_value(expected, value)
        model.group_add_value(group2, value)

        actual.load(group1.dump())
        model.group_merge(actual, group2)
        assert_close(actual.dump(), expected.dump())


@for_each_model()
def test_sample_seed(Model, EXAMPLE):
    model = Model.model_load(EXAMPLE['model'])

    seed_all(0)
    group1 = model.group_create()
    values1 = [model.sample_value(group1) for _ in xrange(DATA_COUNT)]

    seed_all(0)
    group2 = model.group_create()
    values2 = [model.sample_value(group2) for _ in xrange(DATA_COUNT)]

    assert_close(values1, values2, err_msg='values')


@for_each_model()
def test_sample_value(Model, EXAMPLE):
    seed_all(0)
    model = Model.model_load(EXAMPLE['model'])
    for values in [[], EXAMPLE['values']]:
        group = model.group_create(values)
        samples = [model.sample_value(group) for _ in xrange(SAMPLE_COUNT)]
        if Model.Value == int:
            probs_dict = {
                value: math.exp(model.score_value(group, value))
                for value in set(samples)
            }
            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        elif Model.Value == float:
            probs = numpy.exp([
                model.score_value(group, value)
                for value in samples
            ])
            gof = density_goodness_of_fit(samples, probs, plot=True)
        else:
            raise SkipTest('Not implemented for {}'.format(Model.Value))
        print '{} gof = {:0.3g}'.format(Model.__name__, gof)
        assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_model()
def test_sample_group(Model, EXAMPLE):
    seed_all(0)
    SIZE = 2
    model = Model.model_load(EXAMPLE['model'])
    for values in [[], EXAMPLE['values']]:
        if Model.Value == int:
            samples = []
            probs_dict = {}
            for _ in xrange(SAMPLE_COUNT):
                values = model.sample_group(SIZE)
                sample = tuple(values)
                samples.append(sample)
                group = model.group_create(values)
                probs_dict[sample] = math.exp(model.score_group(group))
            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
        else:
            raise SkipTest('Not implemented for {}'.format(Model.Value))
        print '{} gof = {:0.3g}'.format(Model.__name__, gof)
        assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_model(lambda Model: hasattr(Model, 'scorer_create'))
def test_scorer(Model, EXAMPLE):
    model = Model.model_load(EXAMPLE['model'])
    values = EXAMPLE['values']

    group = model.group_create()
    scorer1 = model.scorer_create()
    scorer2 = model.scorer_create(group)
    for value in values:
        score1 = model.scorer_eval(scorer1, value)
        score2 = model.scorer_eval(scorer2, value)
        score3 = model.score_value(group, value)
        assert_all_close([score1, score2, score3])


@for_each_model(lambda Model: hasattr(Model, 'Classifier'))
def test_classifier_runs(Model, EXAMPLE):
    model = Model.model_load(EXAMPLE['model'])
    values = EXAMPLE['values']

    classifier = Model.Classifier()
    for value in values:
        classifier.append(model.group_create([value]))
    model.classifier_init(classifier)

    groupids = []
    for value in values:
        scores = numpy.zeros(len(classifier), dtype=numpy.float32)
        model.classifier_score(classifier, value, scores)
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        model.classifier_add_value(classifier, groupid, value)
        groupids.append(groupid)

    model.classifier_add_group(classifier)
    assert len(classifier) == len(values) + 1
    scores = numpy.zeros(len(classifier), dtype=numpy.float32)

    for value, groupid in zip(values, groupids):
        model.classifier_remove_value(classifier, groupid, value)

    model.classifier_remove_group(classifier, 0)
    model.classifier_remove_group(classifier, len(classifier) - 1)
    assert len(classifier) == len(values) - 1

    for value in values:
        scores = numpy.zeros(len(classifier), dtype=numpy.float32)
        model.classifier_score(classifier, value, scores)
        probs = scores_to_probs(scores)
        groupid = sample_discrete(probs)
        model.classifier_add_value(classifier, groupid, value)


@for_each_model(lambda Model: hasattr(Model, 'Classifier'))
def test_classifier_score(Model, EXAMPLE):
    model = Model.model_load(EXAMPLE['model'])
    values = EXAMPLE['values']

    groups = [model.group_create([value]) for value in values]
    classifier = Model.Classifier()
    for group in groups:
        classifier.append(group)
    model.classifier_init(classifier)

    def check_scores():
        expected = [model.score_value(group, value) for group in groups]
        actual = numpy.zeros(len(classifier), dtype=numpy.float32)
        model.classifier_score(classifier, value, actual)
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
        model.group_add_value(groups[groupid], value)
        model.classifier_add_value(classifier, groupid, value)
        groupids.append(groupid)

    print 'removing'
    for value, groupid in zip(values, groupids):
        model.group_remove_value(groups[groupid], value)
        model.classifier_remove_value(classifier, groupid, value)
        scores = check_scores()
