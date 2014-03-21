from nose.tools import (
    assert_true
)

import distributions.rng
import distributions.hp.random as hpr


def test_rng():
    rng = distributions.rng.Rng()
    assert_true(rng.np is not None)
    assert_true(rng.cc is not None)

    print hpr.random()
    print hpr.random()
    print hpr.random()
    print hpr.random()
