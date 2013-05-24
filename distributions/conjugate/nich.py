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

from scipy.stats import chi2, t, norm
from scipy.special import gammaln
from math import sqrt, log, pi

"""
A conjugate model on normally-distributied univariate data in which the
prior on the mean is normally distributed, and the prior on the variance
is Inverse-Chi-Square distributed.

The equations used here are from Murphy, K. "Conjugate Bayesian
analysis of the Gaussian distribution" (2007)

Equation numbers referenced below are from this paper.
"""


class SS:
    def __init__(self, count, mean, variance):
        self.count = int(count)
        self.mean = float(mean)
        self.variance = float(variance)


class HP:
    def __init__(self, mu, kappa, sigmasq, nu):
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.sigmasq = float(sigmasq)
        self.nu = float(nu)


def create_ss(ss=None, p=None):
    if ss is None:
        return SS(0, 0., 0.)
    else:
        return SS(ss['count'], ss['mean'], ss['variance'])


def dump_ss(ss):
    return vars(ss)


def create_hp(hp=None, p=None):
    if hp is None:
        return HP(0., 1., 1., 1.)
    else:
        return HP(hp['mu'], hp['kappa'], hp['sigmasq'], hp['nu'])


def dump_hp(hp):
    return vars(hp)


def add_data(ss, y):
    ss.count += 1
    delta = y - ss.mean
    ss.mean += delta / ss.count
    ss.variance += delta * (y - ss.mean)


def remove_data(ss, y):
    total = ss.mean * ss.count
    delta = y - ss.mean
    ss.count -= 1
    if ss.count == 0:
        ss.mean = 0.
    else:
        ss.mean = (total - y) / ss.count
    if ss.count <= 1:
        ss.variance = 0.
    else:
        ss.variance -= delta * (y - ss.mean)


def _intermediates(hp, ss):
    """
    Murphy, Eqs.141-144
    """
    total = ss.mean * ss.count
    mu_1 = hp.mu - ss.mean
    kappa_n = hp.kappa + ss.count
    mu_n = (hp.kappa * hp.mu + total) / kappa_n
    nu_n = hp.nu + ss.count
    sigmasq_n = 1. / nu_n * (
        hp.nu * hp.sigmasq
        + ss.variance
        + (ss.count * hp.kappa * mu_1 * mu_1) / kappa_n)
    return HP(mu_n, kappa_n, sigmasq_n, nu_n)


def sample_data(hp, ss):
    (mu, sigmasq) = sample_post(hp, ss)
    return norm.rvs(mu, sqrt(sigmasq))


def sample_post(hp, ss):
    """
    Draw samples from the marginal posteriors of mu and sigmasq

    Murphy, Eqs. 156 & 167
    """
    z = _intermediates(hp, ss)
    # Sample from the inverse-chi^2 using the transform from the chi^2
    sigmasq_star = z.nu * z.sigmasq / chi2.rvs(z.nu)
    mu_star = norm.rvs(z.mu, sqrt(sigmasq_star / z.kappa))

    return (mu_star, sigmasq_star)


def generate_post(hp, ss):
    post = sample_post(hp, ss)
    return {'mu': post[0], 'sigmasq': post[1]}


def log_t_pdf(x, nu, mu, sigmasq):
    """
    Murphy, Eq. 304
    """
    c = gammaln(.5 * (nu + 1.))\
        - (gammaln(.5 * nu) + .5 * (log(nu * pi * sigmasq)))
    xt = (x - mu)
    s = xt * xt / sigmasq
    d = -(.5 * (nu + 1.)) * log(1. + s / nu)
    return c + d


def pred_prob(hp, ss, y):
    """
    Murphy, Eq. 176
    """
    z = _intermediates(hp, ss)
    return log_t_pdf(
        y,
        z.nu,
        z.mu,
        ((1 + z.kappa) * z.sigmasq) / z.kappa)


def data_prob(hp, ss):
    """
    Murphy, Eq. 171
    """
    z = _intermediates(hp, ss)
    return gammaln(z.nu / 2.) - gammaln(hp.nu / 2.) + \
        0.5 * log(hp.kappa / z.kappa) + \
        (0.5 * hp.nu) * log(hp.nu * hp.sigmasq) - \
        (0.5 * z.nu) * log(z.nu * z.sigmasq) - \
        ss.count / 2. * 1.1447298858493991


def add_pred_probs(hp, ss, y, scores):
    size = len(scores)
    assert len(ss) == size
    for i in range(size):
        scores[i] += pred_prob(hp, ss[i], y)
