from nose.plugins.skip import SkipTest

from distributions import ComponentModel

def check_cm(name):
    try:
        ComponentModel(name)
    except KeyError:
        raise SkipTest

