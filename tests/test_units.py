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

from distributions.conjugate import dd, nich
from nose.tools import assert_almost_equal
from math import log


def test_dd_create_ss():
    ss = dd.create_ss()
    assert len(ss.counts) == 2


def test_dd_create_hp():
    hp = dd.create_hp()
    assert len(hp.alphas) == 2


def test_dd_add_remove_data():
    ss = dd.create_ss()
    dd.add_data(ss, 0)
    assert ss.counts[0] == 1
    assert sum(ss.counts) == 1

    dd.remove_data(ss, 0)
    assert ss.counts[0] == 0
    assert sum(ss.counts) == 0


def test_dd_pred_prob():
    ss = dd.create_ss()
    hp = dd.create_hp()
    assert_almost_equal(dd.pred_prob(hp, ss, 0), log(0.5))
    assert_almost_equal(dd.pred_prob(hp, ss, 1), log(0.5))


def test_dd_data_prob():
    ss = dd.create_ss()
    hp = dd.create_hp()
    assert_almost_equal(dd.data_prob(hp, ss), 0)

    dd.add_data(ss, 0)
    assert_almost_equal(dd.data_prob(hp, ss), log(0.5))


def test_dd_sample_data():
    ss = dd.create_ss()
    hp = dd.create_hp()
    dd.sample_data(hp, ss)


def test_dd_sample_post():
    ss = dd.create_ss()
    hp = dd.create_hp()
    dd.sample_post(hp, ss)


def test_nich_create_ss():
    ss = nich.create_ss()
    assert_almost_equal(ss.count, 0)
    assert_almost_equal(ss.mean, 0)
    assert_almost_equal(ss.variance, 0)


def test_nich_create_hp():
    hp = nich.create_hp()
    assert_almost_equal(hp.mu, 0)
    assert_almost_equal(hp.kappa, 1)
    assert_almost_equal(hp.sigmasq, 1.)
    assert_almost_equal(hp.nu, 1.)


def test_nich_add_remove_data():
    ss = nich.create_ss()
    nich.add_data(ss, 0)
    assert_almost_equal(ss.count, 1)
    assert_almost_equal(ss.mean, 0)
    assert_almost_equal(ss.variance, 0)

    nich.remove_data(ss, 0)
    assert_almost_equal(ss.count, 0)
    assert_almost_equal(ss.mean, 0)
    assert_almost_equal(ss.variance, 0)


def test_nich_sample_data():
    ss = nich.create_ss()
    hp = nich.create_hp()
    nich.sample_data(hp, ss)


def test_nich_sample_post():
    ss = nich.create_ss()
    hp = nich.create_hp()
    nich.sample_post(hp, ss)


def test_nich_pred_prob():
    ss = nich.create_ss()
    hp = nich.create_hp()

    assert nich.pred_prob(hp, ss, 0) <= 0


def test_nich_data_prob():
    ss = nich.create_ss()
    hp = nich.create_hp()
    assert_almost_equal(nich.data_prob(hp, ss), 0)
