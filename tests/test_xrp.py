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

from nose.tools import assert_equal, assert_almost_equal
from numpy.random import permutation
from numpy.testing import assert_array_almost_equal

from distributions import ComponentModel
from distributions.util import seed


COUNT = 10
XRPS = [
    'bb',
    'crp',
    'cydd', 'pydd',
    'cydpm', 'pydpm',
    'cygp', 'pygp',
    'ngig',
    'cynich', 'pynich',
    'pyp',
    ]


def check_sample_data_seed(name):
    check_cm(name)
    n = 10
    seed(0)
    cm1 = ComponentModel(name)
    cm1.realize_hp()
    data_values1 = [cm1.sample_data() for _ in range(n)]
    seed(0)
    cm2 = ComponentModel(name)
    cm2.realize_hp()
    data_values2 = [cm2.sample_data() for _ in range(n)]
    for i in range(n):
        assert_almost_equal(data_values1[i], data_values2[i])


def check_sums(name):
    check_cm(name)
    cm = ComponentModel(name)
    cm.realize_hp()
    values = [cm.sample_data() for _ in range(COUNT)]
    score = 0.
    for value in values:
        score += cm.pred_prob(value)
        cm.add_data(value)
    assert_almost_equal(score, cm.data_prob())


def check_exchangeable(name):
    check_cm(name)
    cm = ComponentModel(name)
    cm.realize_hp()
    values = [cm.sample_data() for _ in range(COUNT)]
    p1 = permutation(COUNT)
    p2 = permutation(COUNT)
    for i in range(COUNT):
        cm.add_data(values[p1[i]])
    prob1 = cm.data_prob()
    for i in range(COUNT):
        cm.remove_data(values[p1[i]])
    assert_almost_equal(cm.data_prob(), 0.)
    for i in range(COUNT):
        cm.add_data(values[p2[i]])
    prob2 = cm.data_prob()
    assert_almost_equal(prob1, prob2)


def check_hp_io(name):
    check_cm(name)
    cm = ComponentModel(name)
    cm.realize_hp()
    assert_equal(ComponentModel(name, hp=cm.dump_hp()).dump_hp(), cm.dump_hp())


def check_ss_io(name):
    check_cm(name)
    cm = ComponentModel(name)
    cm.realize_hp()
    assert_equal(ComponentModel(name, ss=cm.dump_ss()).dump_ss(), cm.dump_ss())
    cm.add_data(cm.sample_data())
    assert_equal(ComponentModel(name, ss=cm.dump_ss()).dump_ss(), cm.dump_ss())


def test_xrps():
    for name in XRPS:
        yield check_sample_data_seed, name
        yield check_sums, name
        yield check_exchangeable, name
        yield check_hp_io, name
        yield check_ss_io, name
