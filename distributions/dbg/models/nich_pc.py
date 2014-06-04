# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
A conjugate model on normally-distributied univariate data in which the
prior on the mean is normally distributed, and the prior on the variance
is Inverse-Chi-Square distributed.

The equations used here are from \cite{murphy2007conjugate}.
Murphy, K. "Conjugate Bayesian analysis of the Gaussian distribution" (2007)
Equation numbers referenced below are from this paper.
"""

from distributions.dbg.special import sqrt, log, pi, gammaln
from distributions.dbg.random import sample_chi2, sample_normal, score_normal
from distributions.mixins import GroupIoMixin, SharedIoMixin


ROOT_2_PI = sqrt(2. * pi)

# FIXME how does this relate to distributions.dbg.random.score_student_t
def score_student_t(x, nu, mu, sigmasq):
    """
    \cite{murphy2007conjugate}, Eq. 304
    """
    score = gammaln(.5 * (nu + 1.)) - gammaln(.5 * nu)
    score -= .5 * log(nu * pi * sigmasq)
    xt = (x - mu)
    s = xt * xt / sigmasq
    score += -(.5 * (nu + 1.)) * log(1. + s / nu)
    return score


NAME = 'NormalInverseChiSq'
EXAMPLES = [
    {
        'shared': {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    },
]
Value = float


class Shared(SharedIoMixin):
    def __init__(self):
        self.mu = None
        self.kappa = None
        self.sigmasq = None
        self.nu = None

    def plus_group(self, group):
        total = group.mean * group.count
        kappa_n = self.kappa + group.count
        mu_n = (self.kappa * self.mu + total) / kappa_n

        post = self.__class__()
        post.mu = mu_n
        post.kappa = kappa_n
        post.nu = self.nu
        post.sigmasq = self.sigmasq
        return post

    def load(self, raw):
        self.mu = float(raw['mu'])
        self.kappa = float(raw['kappa'])
        self.sigmasq = float(raw['sigmasq'])
        self.nu = float(raw['nu'])

    def dump(self):
        return {
            'mu': self.mu,
            'kappa': self.kappa,
            'sigmasq': self.sigmasq,
            'nu': self.nu,
        }

    def load_protobuf(self, message):
        self.mu = float(message.mu)
        self.kappa = float(message.kappa)
        self.sigmasq = float(message.sigmasq)
        self.nu = float(message.nu)

    def dump_protobuf(self, message):
        message.Clear()
        message.mu = self.mu
        message.kappa = self.kappa
        message.sigmasq = self.sigmasq
        message.nu = self.nu


class Group(GroupIoMixin):
    def __init__(self):
        # vals + params for nonconj
        self.values = None
        self.sigmasq = None

        # ss for conj
        self.count = None
        self.mean = None
        self.count_times_variance = None  # = count * variance

    def init(self, shared):
        self.values = []
        self.sigmasq = 1.

        self.count = 0
        self.mean = 0.
        self.count_times_variance = 0.

    def add_value(self, shared, value):
        self.values.append(value)

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        if self.count <= 1:
            self.count_times_variance = 0.
        else:
            self.count_times_variance -= delta * (value - self.mean)

    def remove_value(self, shared, value):
        self.values.remove(value)

        total = self.mean * self.count
        delta = value - self.mean
        self.count -= 1
        if self.count == 0:
            self.mean = 0.
        else:
            self.mean = (total - value) / self.count
        if self.count <= 1:
            self.count_times_variance = 0.
        else:
            self.count_times_variance -= delta * (value - self.mean)

    def merge(self, shared, source):
        pass # what to do with params?

    def sample_params(self, shared):
        self.sigmasq = shared.nu * shared.sigmasq / sample_chi2(shared.nu)

    def score_params(self, shared):
        iss = -1. / (2. * self.sigmasq)
        ix_score = iss * shared.nu * shared.sigmasq
        ix_score += log(self.sigmasq ** (1. + shared.nu / 2.))
        return ix_score

    def score_value(self, shared, value):
        '''
        \cite{murphy2007conjugate}, Eq. 36
        '''
        post = shared.plus_group(self)
        return score_normal(
            value, post.mu, self.sigmasq / post.kappa + self.sigmasq)

    def score_data(self, shared):
        '''
        \cite{murphy2007conjugate}, Eqs. 53-55
        '''
        post = shared.plus_group(self)

        sigma = sqrt(self.sigmasq)
        denom = self.sigmasq * post.kappa / shared.kappa
        sumsq = self.count * (self.mean ** 2) - self.count_times_variance

        score = log(sigma / (((ROOT_2_PI * sigma) ** self.count) * sqrt(denom)))
        score -= (sumsq + shared.mu * shared.kappa) / (2. * self.sigmasq)
        score += post.mu ** 2 * (denom / 2.)
        return score

    def score_group(self, shared):
        return self.score_data(shared) + self.score_params(shared)

    def load(self, raw):
        pass

    def dump(self):
        return {}

    def load_protobuf(self, message):
        pass

    def dump_protobuf(self, message):
        pass


def sampler_create(shared, group=None):
    """
    Draw samples from the marginal posteriors of mu and sigmasq

    \cite{murphy2007conjugate}, Eqs. 156 & 167
    """
    post = shared if group is None else shared.plus_group(group)
    # Sample from the inverse-chi^2 using the transform from the chi^2
    sigmasq_star = post.nu * post.sigmasq / sample_chi2(post.nu)
    mu_star = sample_normal(post.mu, sqrt(sigmasq_star / post.kappa))
    return (mu_star, sigmasq_star)


def sampler_eval(shared, sampler):
    mu, sigmasq = sampler
    return sample_normal(mu, sqrt(sigmasq))


def sample_value(shared, group):
    sampler = sampler_create(shared, group)
    return sampler_eval(shared, sampler)


def sample_group(shared, size):
    sampler = sampler_create(shared)
    return [sampler_eval(shared, sampler) for _ in xrange(size)]
