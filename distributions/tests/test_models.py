import os
import glob
import importlib
from nose.tools import assert_true, assert_in, assert_equal


def assert_hasattr(thing, attr):
    assert_true(
        hasattr(thing, attr),
        "{} is missing attribute '{}'".format(thing.__name__, attr))


def discover_modules():
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    models = os.path.join(root, 'models')
    for path in glob.glob(os.path.join(models, '*.p*')):
        filename = os.path.split(path)[-1]
        name = os.path.splitext(filename)[0]
        if not name.startswith('__'):
            yield name


def test_interface():
    for name in discover_modules():
        yield _test_interface, name


def _test_interface(name):
    module_name = 'distributions.models.{}'.format(name)
    module = importlib.import_module(module_name)
    assert_hasattr(module, 'Model')
    Model = module.Model

    assert_hasattr(Model, 'EXAMPLE')
    EXAMPLE = Model.EXAMPLE
    assert_in('model', EXAMPLE)
    assert_in('values', EXAMPLE)
    model = Model.load_model(EXAMPLE['model'])
    values = EXAMPLE['values']
    group1 = model.Group()
    group2 = model.Group()

    model.group_init(group1)
    model.group_init(group2)
    for value in values:
        model.group_add_data(group1, value)
        model.group_add_data(group2, value)
    for value in values:
        model.group_remove_data(group2, value)
    model.group_merge(group2, group1)

    for value in values:
        model.score_value(group1, value)
    for _ in xrange(10):
        value = model.sample_value(group1)
        model.score_value(group1, value)
        model.sample_group(10)
    model.score_group(group1)
    model.score_group(group2)

    assert_equal(model.dump(), EXAMPLE['model'])
    assert_equal(model.dump(), Model.dump_model(model))
    assert_equal(group1.dump(), Model.dump_group(group1))
