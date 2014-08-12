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
A conjugate self on normally-distributied univariate data in which the
prior on the mean is normally distributed, and the prior on the variance
is Inverse-Chi-Square distributed.

The equations used here are from \cite{murphy2007conjugate}
Murphy, K. "Conjugate Bayesian analysis of the Gaussian distribution" (2007)
Equation numbers referenced below are from this paper.
"""

import numpy
cimport numpy
numpy.import_array()
from distributions.hp.special cimport sqrt, log, gammaln, M_PI
from distributions.hp.random cimport sample_normal, sample_chisq
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


# scalar score_student_t, see distributions.dbg.random.score_student_t
# for the multivariate generalization
cdef double score_student_t(double x, double nu, double mu, double sigmasq):
    """
    \cite{murphy2007conjugate}, Eq. 304
    """
    cdef double c = gammaln(.5 * (nu + 1.))\
        - (gammaln(.5 * nu) + .5 * (log(nu * M_PI * sigmasq)))
    cdef double xt = (x - mu)
    cdef double s = xt * xt / sigmasq
    cdef double d = -(.5 * (nu + 1.)) * log(1. + s / nu)
    return c + d


NAME = 'NormalInverseChiSq'
EXAMPLES = [
    {
        'shared': {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    },
]
Value = float


ctypedef double _Value


cdef class _Shared:
    cdef double mu
    cdef double kappa
    cdef double sigmasq
    cdef double nu

    cdef _Shared plus_group(self, _Group group):
        """
        \cite{murphy2007conjugate}, Eqs.141-144
        """
        cdef double mu_1 = self.mu - group.mean
        cdef double kappa_n = self.kappa + group.count
        cdef double mu_n = (
            self.kappa * self.mu + group.mean * group.count) / kappa_n
        cdef double nu_n = self.nu + group.count
        cdef double sigmasq_n = 1. / nu_n * (
            self.nu * self.sigmasq
            + group.count_times_variance
            + (group.count * self.kappa * mu_1 * mu_1) / kappa_n)

        cdef _Shared post = _Shared()
        post.mu = mu_n
        post.kappa = kappa_n
        post.nu = nu_n
        post.sigmasq = sigmasq_n
        return post

    def load(self, dict raw):
        self.mu = raw['mu']
        self.kappa = raw['kappa']
        self.sigmasq = raw['sigmasq']
        self.nu = raw['nu']

    def dump(self):
        return {
            'mu': self.mu,
            'kappa': self.kappa,
            'sigmasq': self.sigmasq,
            'nu': self.nu,
        }


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group:
    cdef size_t count
    cdef double mean
    cdef double count_times_variance

    def init(self, _Shared shared):
        self.count = 0
        self.mean = 0.
        self.count_times_variance = 0.

    def add_value(self, _Shared shared, double value):
        self.count += 1
        cdef double delta = value - self.mean
        self.mean += delta / self.count
        self.count_times_variance += delta * (value - self.mean)

    def add_repeated_value(self, _Shared shared, double value, int count):
        self.count += count
        cdef double delta = count * value - self.mean
        self.mean += delta / self.count
        self.count_times_variance += delta * (value - self.mean)

    def remove_value(self, _Shared shared, double value):
        cdef double total = self.mean * self.count
        cdef double delta = value - self.mean
        self.count -= 1
        if self.count == 0:
            self.mean = 0.
        else:
            self.mean = (total - value) / self.count
        if self.count <= 1:
            self.count_times_variance = 0.
        else:
            self.count_times_variance -= delta * (value - self.mean)

    def merge(self, _Shared shared, _Group source):
        cdef size_t count = self.count + source.count
        cdef double delta = source.mean - self.mean
        cdef double source_part = <double> source.count / count
        cdef double cross_part = self.count * source_part
        self.count = count
        self.mean += source_part * delta
        self.count_times_variance += \
            source.count_times_variance + cross_part * delta * delta

    def score_value(self, _Shared shared, _Value value):
        """
        \cite{murphy2007conjugate}, Eq. 176
        """
        cdef _Shared post = shared.plus_group(self)
        return score_student_t(
            value,
            post.nu,
            post.mu,
            ((1 + post.kappa) * post.sigmasq) / post.kappa)

    def score_data(self, _Shared shared):
        """
        \cite{murphy2007conjugate}, Eq. 171
        """
        cdef _Shared post = shared.plus_group(self)
        return gammaln(post.nu / 2.) - gammaln(shared.nu / 2.) + \
            0.5 * log(shared.kappa / post.kappa) + \
            (0.5 * shared.nu) * log(shared.nu * shared.sigmasq) - \
            (0.5 * post.nu) * log(post.nu * post.sigmasq) - \
            self.count / 2. * 1.1447298858493991

    def sample_value(self, _Shared shared):
        cdef Sampler sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, dict raw):
        self.count = raw['count']
        self.mean = raw['mean']
        self.count_times_variance = raw['count_times_variance']

    def dump(self):
        return {
            'count': self.count,
            'mean': self.mean,
            'count_times_variance': self.count_times_variance,
        }


class Group(_Group, GroupIoMixin):
    pass


cdef class Sampler:
    cdef double mu
    cdef double sigmasq

    def init(self, _Shared shared, _Group group=None):
        """
        Draw samples from the marginal posteriors of mu and sigmasq

        \cite{murphy2007conjugate}, Eqs. 156 & 167
        """
        cdef _Shared post
        post = shared if group is None else shared.plus_group(group)
        self.sigmasq = post.nu * post.sigmasq / sample_chisq(post.nu)
        self.mu = sample_normal(post.mu, self.sigmasq / post.kappa)

    def eval(self, _Shared shared):
        return sample_normal(self.mu, self.sigmasq)


def sample_group(_Shared shared, int size):
    cdef Sampler sampler = Sampler()
    sampler.init(shared)
    cdef list result = []
    cdef int i
    for i in xrange(size):
        result.append(sampler.eval(shared))
    return result
