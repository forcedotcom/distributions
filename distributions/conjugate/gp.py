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

from numpy.random import gamma, poisson
from scipy.special import gammaln
from math import log, factorial


class SS:
    def __init__(self, n, sum, log_prod):
        self.n = int(n)
        self.sum = int(sum)
        self.log_prod = float(log_prod)


class HP:
    def __init__(self, alpha, beta):
        self.alpha = float(alpha)
        self.beta = float(beta)


def create_ss(ss=None, p=None):
    if ss is None:
        return SS(0, 0, 0.)
    else:
        return SS(ss['n'], ss['sum'], ss['log_prod'])


def dump_ss(ss):
    return vars(ss)


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(1., 1.)
    else:
        return HP(hp['alpha'], hp['beta'])


def dump_hp(hp):
    return vars(hp)


def add_data(ss, y):
    ss.n += 1
    ss.sum += int(y)
    ss.log_prod += log(factorial(y))


def remove_data(ss, y):
    ss.n -= 1
    ss.sum -= int(y)
    ss.log_prod -= log(factorial(y))


def _intermediates(hp, ss):
    alpha_n = hp.alpha + ss.sum
    beta_n = 1. / (ss.n + 1. / hp.beta)
    return HP(alpha_n, beta_n)


def sample_post(hp, ss):
    z = _intermediates(hp, ss)
    return gamma(z.alpha, z.beta)


def generate_post(hp, ss):
    return {'lambda': sample_post(hp, ss)}


def sample_data(hp, ss):
    mu = sample_post(hp, ss)
    return poisson(mu)


def pred_prob(hp, ss, y):
    z = _intermediates(hp, ss)
    return gammaln(z.alpha + y) - gammaln(z.alpha) - \
        z.alpha * log(z.beta) + \
        (z.alpha + y) * log(1. / (1. + 1. / z.beta)) - \
        log(factorial(y))


def data_prob(hp, ss):
    z = _intermediates(hp, ss)
    return gammaln(z.alpha) - gammaln(hp.alpha) + \
        z.alpha * log(z.beta) - hp.alpha * log(hp.beta) - \
        ss.log_prod


def add_pred_probs(hp, ss, y, scores):
    size = len(scores)
    assert len(ss) == size
    for i in range(size):
        scores[i] += pred_prob(hp, ss[i], y)
