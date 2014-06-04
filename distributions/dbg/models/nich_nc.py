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
        post = self.__class__()
        post.mu = self.mu
        post.kappa = self.kappa
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
        self.values = None
        self.mu = None
        self.sigmasq = None

    def init(self, shared):
        self.values = []
        self.mu = 0.
        self.sigmasq = 1.

    def add_value(self, shared, value):
        self.values.append(value)

    def remove_value(self, shared, value):
        self.values.remove(value)

    def merge(self, shared, source):
        self.values.extend(source.values)

    def sample_params(self, shared):
        self.sigmasq = shared.nu * shared.sigmasq / sample_chi2(shared.nu)
        self.mu = sample_normal(shared.mu, sqrt(self.sigmasq / shared.kappa))

    def score_params(self, shared):
        # proportional to Murphy 125
        iss = -1. / (2. * self.sigmasq)
        n_score = iss * shared.kappa * (self.mu - shared.mu) ** 2
        n_score += log(1. / sqrt(self.sigmasq))
        ix_score = iss * shared.nu * shared.sigmasq
        ix_score += log(self.sigmasq ** (1. + shared.nu / 2.))
        return n_score + ix_score

    def score_value(self, shared, value):
        return score_normal(value, self.mu, self.sigmasq)

    def score_data(self, shared):
        return sum(self.score_value(shared, value) for value in self.values)

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
    if group is None:
        group = Group()
        group.sample_params(shared)
    return (group.mu, group.sigmasq)


def sampler_eval(shared, sampler):
    mu, sigmasq = sampler
    return sample_normal(mu, sqrt(sigmasq))


def sample_value(shared, group):
    sampler = sampler_create(shared, group)
    return sampler_eval(shared, sampler)


def sample_group(shared, size):
    sampler = sampler_create(shared)
    return [sampler_eval(shared, sampler) for _ in xrange(size)]
