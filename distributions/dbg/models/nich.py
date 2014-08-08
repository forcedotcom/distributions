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
from distributions.dbg.random import sample_chi2, sample_normal
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


# scalar score_student_t, see distributions.dbg.random.score_student_t
# for the multivariate generalization
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


class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self.mu = None
        self.kappa = None
        self.sigmasq = None
        self.nu = None

    def plus_group(self, group):
        """
        \cite{murphy2007conjugate}, Eqs.141-144
        """
        total = group.mean * group.count
        mu_1 = self.mu - group.mean
        kappa_n = self.kappa + group.count
        mu_n = (self.kappa * self.mu + total) / kappa_n
        nu_n = self.nu + group.count
        sigmasq_n = 1. / nu_n * (
            self.nu * self.sigmasq
            + group.count_times_variance
            + (group.count * self.kappa * mu_1 * mu_1) / kappa_n)

        post = self.__class__()
        post.mu = mu_n
        post.kappa = kappa_n
        post.nu = nu_n
        post.sigmasq = sigmasq_n
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

    def protobuf_load(self, message):
        self.mu = float(message.mu)
        self.kappa = float(message.kappa)
        self.sigmasq = float(message.sigmasq)
        self.nu = float(message.nu)

    def protobuf_dump(self, message):
        message.Clear()
        message.mu = self.mu
        message.kappa = self.kappa
        message.sigmasq = self.sigmasq
        message.nu = self.nu


class Group(GroupIoMixin):
    def __init__(self):
        self.count = None
        self.mean = None
        self.count_times_variance = None  # = count * variance

    def init(self, shared):
        self.count = 0
        self.mean = 0.
        self.count_times_variance = 0.

    def add_value(self, shared, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.count_times_variance += delta * (value - self.mean)

    def add_repeated_value(self, shared, value, count):
        self.count += count
        delta = count * value - self.mean
        self.mean += delta / self.count
        self.count_times_variance += delta * (value - self.mean)

    def remove_value(self, shared, value):
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
        count = self.count + source.count
        delta = source.mean - self.mean
        source_part = float(source.count) / count
        cross_part = self.count * source_part
        self.count = count
        self.mean += source_part * delta
        self.count_times_variance += \
            source.count_times_variance + cross_part * delta * delta

    def score_value(self, shared, value):
        """
        \cite{murphy2007conjugate}, Eq. 176
        """
        post = shared.plus_group(self)
        return score_student_t(
            value,
            post.nu,
            post.mu,
            ((1 + post.kappa) * post.sigmasq) / post.kappa)

    def score_data(self, shared):
        """
        \cite{murphy2007conjugate}, Eq. 171
        """
        post = shared.plus_group(self)
        return gammaln(post.nu / 2.) - gammaln(shared.nu / 2.) \
            + 0.5 * log(shared.kappa / post.kappa) \
            + (0.5 * shared.nu) * log(shared.nu * shared.sigmasq) \
            - (0.5 * post.nu) * log(post.nu * post.sigmasq) \
            - self.count / 2. * 1.1447298858493991

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, raw):
        self.count = int(raw['count'])
        self.mean = float(raw['mean'])
        self.count_times_variance = float(raw['count_times_variance'])

    def dump(self):
        return {
            'count': self.count,
            'mean': self.mean,
            'count_times_variance': self.count_times_variance,
        }

    def protobuf_load(self, message):
        self.count = int(message.count)
        self.mean = float(message.mean)
        self.count_times_variance = float(message.count_times_variance)

    def protobuf_dump(self, message):
        message.count = self.count
        message.mean = self.mean
        message.count_times_variance = self.count_times_variance


class Sampler(object):
    def init(self, shared, group=None):
        """
        Draw samples from the marginal posteriors of mu and sigmasq

        \cite{murphy2007conjugate}, Eqs. 156 & 167
        """
        post = shared if group is None else shared.plus_group(group)
        # Sample from the inverse-chi^2 using the transform from the chi^2
        sigmasq_star = post.nu * post.sigmasq / sample_chi2(post.nu)
        self.sigma = sqrt(sigmasq_star)
        self.mu = sample_normal(post.mu, sqrt(sigmasq_star / post.kappa))

    def eval(self, shared):
        return sample_normal(self.mu, self.sigma)


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
