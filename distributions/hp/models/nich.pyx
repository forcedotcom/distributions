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
from distributions.mixins import ComponentModel, Serializable


# FIXME how does this relate to distributions.dbg.random.score_student_t
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

#-------------------------------------------------------------------------
# Datatypes

ctypedef double Value


cdef class Group:
    cdef size_t count
    cdef double mean
    cdef double count_times_variance

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


ctypedef tuple Sampler


cdef class Model_cy:

    cdef double mu
    cdef double kappa
    cdef double sigmasq
    cdef double nu

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

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.count = 0
        group.mean = 0.
        group.count_times_variance = 0.

    def group_add_value(self, Group group, double value):
        group.count += 1
        cdef double delta = value - group.mean
        group.mean += delta / group.count
        group.count_times_variance += delta * (value - group.mean)


    def group_remove_value(self, Group group, double value):
        cdef double total = group.mean * group.count
        cdef double delta = value - group.mean
        group.count -= 1
        if group.count == 0:
            group.mean = 0.
        else:
            group.mean = (total - value) / group.count
        if group.count <= 1:
            group.count_times_variance = 0.
        else:
            group.count_times_variance -= delta * (value - group.mean)

    def group_merge(self, Group destin, Group source):
        cdef size_t count = destin.count + source.count
        cdef double delta = source.mean - destin.mean
        cdef double source_part = <double> source.count / count
        cdef double cross_part = destin.count * source_part
        destin.count = count
        destin.mean += source_part * delta
        destin.count_times_variance += \
            source.count_times_variance + cross_part * delta * delta

    cdef Model_cy plus_group(self, Group group):
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

        cdef Model_cy post = Model_cy()
        post.mu = mu_n
        post.kappa = kappa_n
        post.nu = nu_n
        post.sigmasq = sigmasq_n
        return post

    #-------------------------------------------------------------------------
    # Sampling

    cpdef Sampler sampler_create(self, Group group=None):
        """
        Draw samples from the marginal posteriors of mu and sigmasq

        \cite{murphy2007conjugate}, Eqs. 156 & 167
        """
        cdef Model_cy post = self if group is None else self.plus_group(group)
        cdef double sigmasq_star = post.nu * post.sigmasq / sample_chisq(post.nu)
        cdef double mu_star = sample_normal(post.mu, sigmasq_star / post.kappa)
        return (mu_star, sigmasq_star)

    cpdef Value sampler_eval(self, Sampler sampler):
        cdef double mu = sampler[0]
        cdef double sigmasq = sampler[1]
        return sample_normal(mu, sigmasq)

    def sample_value(self, Group group):
        cdef Sampler sampler = self.sampler_create(group)
        return self.sampler_eval(sampler)

    def sample_group(self, int size):
        cdef Sampler sampler = self.sampler_create()
        cdef list result = []
        cdef int i
        for i in xrange(size):
            result.append(self.sampler_eval(sampler))
        return result

    #-------------------------------------------------------------------------
    # Scoring

    cpdef double score_value(self, Group group, Value value):
        """
        \cite{murphy2007conjugate}, Eq. 176
        """
        cdef Model_cy post = self.plus_group(group)
        return score_student_t(
            value,
            post.nu,
            post.mu,
            ((1 + post.kappa) * post.sigmasq) / post.kappa)

    def score_group(self, Group group):
        """
        \cite{murphy2007conjugate}, Eq. 171
        """
        cdef Model_cy post = self.plus_group(group)
        return gammaln(post.nu / 2.) - gammaln(self.nu / 2.) + \
            0.5 * log(self.kappa / post.kappa) + \
            (0.5 * self.nu) * log(self.nu * self.sigmasq) - \
            (0.5 * post.nu) * log(post.nu * post.sigmasq) - \
            group.count / 2. * 1.1447298858493991

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
            'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
        },
    ]


class NormalInverseChiSq(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = float

    Group = Group


Model = NormalInverseChiSq
