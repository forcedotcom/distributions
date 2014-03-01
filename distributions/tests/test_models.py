import os
import glob
import importlib
from nose.tools import assert_true


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
    #Model = module.Model
