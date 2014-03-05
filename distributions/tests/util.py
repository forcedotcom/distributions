import os
import glob
from nose.tools import assert_true, assert_less, assert_equal
import numpy
from numpy.testing import assert_array_almost_equal
import importlib
import distributions.dbg.random
import distributions.hp.random
#import distributions.lp.random

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def seed_all(s):
    distributions.dbg.random.seed(s)
    distributions.hp.random.seed(s)
    #distributions.lp.random.seed(s)


def list_models():
    for path in glob.glob(os.path.join(ROOT, '*', 'models', '*.p*')):
        dirname, filename = os.path.split(path)
        flavor = os.path.split(os.path.dirname(dirname))[-1]
        name = os.path.splitext(filename)[0]
        if not name.startswith('__'):
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
        assert_equal(set(lhs.keys()), set(rhs.keys()))
        for key, val in lhs.iteritems():
            msg = '{}[{}]'.format(err_msg or '', key)
            assert_close(val, rhs[key], percent, tol, msg)
    elif isinstance(lhs, float) or isinstance(lhs, numpy.float64):
        assert isinstance(rhs, float)
        diff = abs(lhs - rhs)
        norm = (abs(lhs) + abs(rhs)) * (percent / 100) + tol
        msg = '{} off by {}% = {}'.format(err_msg, 100 * diff / norm, diff)
        assert_less(diff, norm, msg)
    elif isinstance(lhs, numpy.ndarray) or isinstance(lhs, list):
        assert_array_almost_equal(lhs, rhs, err_msg=(err_msg or ''))
    else:
        assert_equal(lhs, rhs, err_msg)


def assert_all_close(collection, **kwargs):
    for i1, item1 in enumerate(collection[:-1]):
        for item2 in collection[i1 + 1:]:
            assert_close(item1, item2, **kwargs)
