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

from math import sqrt, pi, log
from numpy.random import gamma, wald
from scipy.stats import t
from scipy.special import gammaln


class SS:
    def __init__(self, n, sum, r_sum, log_prod):
        self.n = int(n)
        self.sum = float(sum)
        self.r_sum = float(r_sum)
        self.log_prod = float(log_prod)


class HP:
    def __init__(self, alpha, beta, mu, tau):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mu = float(mu)
        self.tau = float(tau)


def create_ss(ss=None, p=None):
    if ss is None:
        return SS(0, 0., 0., 0.)
    else:
        return SS(ss['n'], ss['sum'], ss['r_sum'], ss['log_prod'])


def dump_ss(ss):
    return vars(ss)


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(1., 1., 1., 1.)
    else:
        return HP(hp['alpha'], hp['beta'], hp['mu'], hp['tau'])


def dump_hp(hp):
    return vars(hp)


def add_data(ss, y):
    y = float(y)
    ss.n += 1
    ss.sum += y
    ss.r_sum += 1. / y
    ss.log_prod += 3 * log(y)


def remove_data(ss, y):
    y = float(y)
    ss.n -= 1
    ss.sum -= y
    ss.r_sum -= 1. / y
    ss.log_prod -= 3 * log(y)


def _intermediates(hp, ss):
    alpha_n = hp.alpha + ss.n / 2.
    beta_n = hp.beta \
            + (hp.tau * hp.mu ** 2 + ss.r_sum) / 2. \
            - ((hp.mu * hp.tau + ss.n) ** 2.) / \
            (2. * (hp.tau + ss.sum))
    mu_n = (hp.tau * hp.mu + ss.n) / (hp.tau + ss.sum)
    tau_n = hp.tau + ss.sum
    return HP(alpha_n, beta_n, mu_n, tau_n)


def sample_data(hp, ss):
    (m, l) = sample_post(hp, ss)
    return wald(m, l)


def sample_post(hp, ss):
    z = _intermediates(hp, ss)
    l_star = gamma(z.alpha, 1. / z.beta)
    while True:
        m_star = t.rvs(2 * z.alpha, z.mu,
                       z.beta / (z.alpha * z.tau)) ** -1
        if m_star > 0:
            break
    return (m_star, l_star)


def generate_post(hp, ss):
    mu, lambda_ = sample_post(hp, ss)
    return {'mu': mu, 'lambda': lambda_}


def pred_prob(hp, ss, y):
    z = _intermediates(hp, ss)
    return + z.alpha * log(z.beta) \
            - (z.alpha + 0.5) * log(z.beta + \
                ((z.mu ** 2) * z.tau + 1. / y) / 2. \
                - ((z.mu * z.tau + 1) ** 2) / (2 * (z.tau + y))) \
            + gammaln(z.alpha + 0.5) - gammaln(z.alpha) \
            + 0.5 * log(z.tau / (z.tau + y)) \
            - 0.5 * log(2 * pi * y ** 3)


def data_prob(hp, ss):
    z = _intermediates(hp, ss)
    return + hp.alpha * log(hp.beta) - z.alpha * log(z.beta) \
            + gammaln(z.alpha) - gammaln(hp.alpha) \
            + 0.5 * log(hp.tau / z.tau) \
            - (ss.n * 0.5) * log(2 * pi) - (0.5) * ss.log_prod
