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

from scipy.special import multigammaln
from math import log
import numpy.random
import numpy as np
import numpy.linalg as linalg

from distributions.util import student_t_sample, log_student_t, \
    wishart_sample2


DEFAULT_DIMENSIONS = 2


class SS:
    def __init__(self, n, sum, sumsq):
        self.n = n
        self.sum = sum
        self.sumsq = sumsq


class HP:
    def __init__(self, nu, mu, kappa, Lambda):
        self.nu = nu
        self.mu = mu
        self.kappa = kappa
        self.Lambda = Lambda


def create_ss(ss=None, p=None):
    if ss is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return SS(0, np.zeros(D), np.zeros((D, D)))
    else:
        return SS(ss['n'], np.array(ss['sum']), np.array(ss['sumsq']))


def dump_ss(ss):
    return {
        'n': ss.n,
        'sum': ss.sum.tolist(),
        'sumsq': [row.tolist() for row in ss.sumsq],
        }


def create_hp(hp=None, p=None):
    if hp is None:
        if p is None:
            p = {}
        D = p.get('D', DEFAULT_DIMENSIONS)
        return HP(D, np.zeros(D), 1., np.identity(D))
    else:
        return HP(
            hp['nu'], np.array(hp['mu']), hp['kappa'], np.array(hp['Lambda']))


def dump_hp(hp):
    return {
        'nu': hp.nu,
        'mu': hp.mu.tolist(),
        'kappa': hp.kappa,
        'Lambda': [row.tolist() for row in hp.Lambda]
        }


def add_data(ss, y):
    ss.n += 1
    ss.sum += y
    ss.sumsq += np.outer(y, y)


def remove_data(ss, y):
    ss.n -= 1
    ss.sum -= y
    ss.sumsq -= np.outer(y, y)


def sample_data(hp, ss):
    mu = sample_post_mu(hp, ss)
    sigma = sample_post_sigma(hp, ss)
    return numpy.random.multivariate_normal(mu, sigma)


def sample_post(hp, ss):
    mu = sample_post_mu(hp, ss)
    Sigma = sample_post_sigma(hp, ss)
    return mu, Sigma


def sample_post_mu(hp, ss):
    """
    Murphy, Eqs. 256
    """
    z = _intermediates(hp, ss)
    d = len(ss.sum)
    S = z.Lambda / (z.kappa * (z.nu - d + 1))
    return student_t_sample(z.nu - d + 1, z.mu, S)


def generate_post(hp, ss):
    mu, Sigma = sample_post(hp, ss)
    return {
        'mu': mu.tolist(),
        'Sigma': [row.tolist() for row in Sigma]
        }


def sample_post_sigma(hp, ss):
    """
    Murphy, Eqs. 255
    """
    z = _intermediates(hp, ss)
    return linalg.inv(wishart_sample2(z.nu, linalg.inv(z.Lambda)))


def pred_prob(hp, ss, y):
    """
    Murphy, Eq. 258
    """
    z = _intermediates(hp, ss)
    d = len(ss.sum)
    S = z.Lambda * (z.kappa + 1) \
        / (z.kappa * (z.nu - d + 1))
    return log_student_t(y, z.nu - d + 1, z.mu, S)


def data_prob(hp, ss):
    """
    Murphy, Eq. 266
    """
    z = _intermediates(hp, ss)
    d = len(ss.sum)
    return -0.5 * ss.n * d * 1.1447298858493991 \
        + multigammaln(0.5 * z.nu, d) \
        - multigammaln(0.5 * hp.nu, d) \
            + 0.5 * hp.nu * log(linalg.det(hp.Lambda)) \
        - 0.5 * z.nu * log(linalg.det(z.Lambda)) \
        + 0.5 * d * log(hp.kappa / z.kappa)


def _intermediates(hp, ss):
    """
    Murphy, Eqs. 251-254

    Note: the ybar in Eq. 251 is a typo; should say xbar
    """
    if ss.n == 0:
        xbar = 0.
        S = 0.
    else:
        xbar = ss.sum / ss.n
        # The scatter matrix:
        S = ss.sumsq - np.outer(ss.sum, ss.sum) / ss.n

    kappa_n = hp.kappa + ss.n
    nu_n = hp.nu + ss.n
    mu_n = hp.kappa * hp.mu / (hp.kappa + ss.n) \
        + ss.n * xbar / (hp.kappa + ss.n)
    z = (xbar - hp.mu)
    Lambda_n = hp.Lambda + S \
        + (hp.kappa * ss.n) / (hp.kappa + ss.n) * (np.outer(z, z))
    return HP(nu_n, mu_n, kappa_n, Lambda_n)
