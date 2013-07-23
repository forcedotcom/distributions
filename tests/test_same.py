# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from nose.tools import assert_equal, assert_almost_equal, assert_set_equal
from distributions import ComponentModel
from util import check_cm


DPS = 100


def assert_close(x, y, places=8):
    assert_equal(type(x), type(y))
    if isinstance(x, dict):
        assert_set_equal(set(x.iterkeys()), set(y.iterkeys()))
        for key in x.iterkeys():
            assert_close(x[key], y[key], places)
    elif isinstance(x, list) or isinstance(x, tuple):
        assert_equal(len(x), len(y))
        for xi, yi in zip(x, y):
            assert_close(xi, yi, places)
    else:
        assert_almost_equal(x, y, places)


def check_hp(a, b):
    check_cm(a)
    check_cm(b)
    a = ComponentModel(a)
    b = ComponentModel(b)
    assert_equal(a.dump_hp(), b.dump_hp())


def check_ss(a, b):
    check_cm(a)
    check_cm(b)
    a = ComponentModel(a)
    a.realize_hp()
    b = ComponentModel(b, hp=a.dump_hp())
    dps = [a.sample_data() for _ in range(DPS)]
    assert_equal(a.dump_ss(), b.dump_ss())
    for y in dps:
        a.add_data(y)
        b.add_data(y)
        assert_close(a.dump_ss(), b.dump_ss())
    for y in dps:
        a.remove_data(y)
        b.remove_data(y)
        assert_close(a.dump_ss(), b.dump_ss())


def check_probs(a, b):
    check_cm(a)
    check_cm(b)
    a = ComponentModel(a)
    a.realize_hp()
    b = ComponentModel(b, hp=a.dump_hp())
    dps = [a.sample_data() for _ in range(DPS)]
    for y in dps:
        assert_almost_equal(a.data_prob(), b.data_prob())
        assert_almost_equal(a.pred_prob(y), b.pred_prob(y))
        a.add_data(y)
        b.add_data(y)


PAIRS = [
    ('pynich', 'pynich'),
    ('cynich', 'cynich'),
    ('cynich', 'pynich'),
    ('pydd', 'pydd'),
    ('cydd', 'cydd'),
    ('cydd', 'pydd'),
    ('pydpm', 'pydpm'),
    ('cydpm', 'cydpm'),
    ('cydpm', 'pydpm'),
    ('pygp', 'pygp'),
    ('cygp', 'cygp'),
    ('cygp', 'pygp'),
    ]


def test_same():
    for a, b in PAIRS:
        yield check_hp, a, b
        yield check_ss, a, b
        yield check_probs, a, b
