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

from nose.tools import *
import numpy as np
from numpy.testing import assert_array_almost_equal
import itertools
import random


from distributions import BasicDistribution, ComponentModel
from distributions.util import seed
from distributions.summaries import summarize
from distributions.conjugate import nich, dd, gp, dpm
from util import check_cm


COUNT = 10
CONJUGATES = [
    'cydd', 'pydd',
    'cydpm', 'pydpm',
    'cygp', 'pygp',
    'ngig',
    'cynich', 'pynich',
    ]


def check_summarize(name):
    check_cm(name)
    cm = ComponentModel(name)
    x = []
    for _ in range(COUNT):
        x.append(cm.sample_data())
        summarize(name, x)


def check_summarize_N(name):
    check_cm(name)
    cm = ComponentModel(name)
    x = cm.sample_data(COUNT)
    summarize(name, x)


def check_sample_post_seed(name):
    check_cm(name)
    seed(0)
    cm1 = ComponentModel(name)
    post_values1 = [cm1.sample_post() for _ in range(COUNT)]
    seed(0)
    cm2 = ComponentModel(name)
    post_values2 = [cm2.sample_post() for _ in range(COUNT)]
    for i in range(COUNT):
        assert_array_almost_equal(post_values1[i], post_values2[i])


def check_generate(name):
    check_cm(name)
    cm = ComponentModel(name)
    cm.realize_hp()
    params = cm.generate_post()
    b = BasicDistribution(name, pm=params)
    b.sample_data()


def test_conjugate():
    for name in CONJUGATES:
        yield check_summarize, name
        yield check_generate, name
        yield check_sample_post_seed, name


def assert_close(lhs, rhs, percent=0.1, tol=1e-3, err_msg=None):
    if isinstance(lhs, float) or isinstance(lhs, np.float64):
        assert isinstance(rhs, float)
        diff = abs(lhs - rhs)
        norm = (abs(lhs) + abs(rhs)) * (percent / 100) + tol
        msg = '%s off by %s%% = %s' % (err_msg, 100 * diff / norm, diff)
        assert_less(diff, norm, msg)
    elif isinstance(lhs, int):
        assert_equal(lhs, rhs, err_msg)
    elif isinstance(lhs, np.ndarray):
        assert_array_almost_equal(lhs, rhs, err_msg=(err_msg or ''))
    elif isinstance(lhs, dict):
        assert_equal(lhs, rhs, err_msg)
    elif lhs.__class__.__name__ in ['SS', 'HP']:
        for key in dir(lhs):
            if key[0] != '_':
                lhs_val = getattr(lhs, key)
                rhs_val = getattr(rhs, key)
                msg = '%s (bad %s.%s)' % (err_msg, lhs.__class__.__name__, key)
                assert_close(lhs_val, rhs_val, percent, err_msg=msg)
    else:
        print lhs, rhs
        raise TypeError()


def add_remove_add(name, raw_hps, raw_ss0=None):
    '''
    This tests add_data, remove_data, pred_prob, data_prob
    '''

    DATA_COUNT = 20

    for raw_hp in raw_hps:

        cm = ComponentModel(name, hp=raw_hp, ss=raw_ss0)
        cm.realize_hp()
        data = []
        score = 0

        for _ in range(DATA_COUNT):
            dp = cm.sample_data()
            data.append(dp)
            score += cm.pred_prob(dp)
            cm.add_data(dp)

        cm_all = ComponentModel(name, ss=cm.dump_ss())
        assert_close(
                score,
                cm.data_prob(),
                err_msg='p(x1,...,xn) != p(x1) p(x2|x1) p(xn|...)')

        random.shuffle(data)

        for dp in data:
            cm.remove_data(dp)

        cm0 = ComponentModel(name, ss=raw_ss0)
        assert_close(cm.ss, cm0.ss, err_msg='ss + data - data != ss')

        random.shuffle(data)

        for dp in data:
            cm.add_data(dp)

        assert_close(cm.ss, cm_all.ss, err_msg='ss - data + data != ss')


def test_nich():

    mus = [-10.0, -1.0, 0.0, 1.0, 10.0]
    kappas = [0.5, 1.0, 2.0, 4.0]
    sigmasqs = [0.5, 1.0, 2.0, 4.0]
    nus = [0.5, 1.0, 2.0, 4.0]

    hps = itertools.product(mus, kappas, sigmasqs, nus)
    hps = [dict(zip(('mu', 'kappa', 'sigmasq', 'nu'), hp)) for hp in hps]

    add_remove_add('NICH', hps)


def test_add(dim=20):

    assert dim <= 256

    hps = [{'alphas': [random.uniform(0.1, 4) for _ in range(dim)]}
           for _ in range(10)]

    ss = {'counts': np.zeros(dim)}

    add_remove_add('DD', hps, ss)


def test_gp():

    alphas = [0.5, 1.0, 2.0, 4.0]
    betas = [0.5, 1.0, 2.0, 4.0]

    hps = itertools.product(alphas, betas)
    hps = [dict(zip(('alpha', 'beta'), hp)) for hp in hps]

    add_remove_add('GP', hps)


def test_dpm():

    alphas = [0.1, 1., 10., 20.]
    betas = [{'0': 0.4,
              '1': 0.4},
             {'0': 0.5,
              '1': 0.5},
             {'0': 0.85,
              '1': 0.05},
             {'0': 0.9,
              '1': 0.1}]
    gammas = [1.]
    hps = itertools.product(alphas, betas, gammas)
    hps = [dict(zip(('alpha', 'betas', 'gamma'), hp)) for hp in hps]
    for hp in hps:
        hp['beta0'] = 1 - sum(hp['betas'].itervalues())

    add_remove_add('DPM', hps)
