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

import numpy
cimport numpy
numpy.import_array()
from distributions.hp.special cimport sqrt, log, gammaln, log_factorial
from distributions.hp.random cimport sample_poisson, sample_gamma
from distributions.mixins import ComponentModel, Serializable


ctypedef int Value


cdef class Group:
    cdef int count
    cdef int sum
    cdef double log_prod

    def load(self, raw):
        self.count = raw['count']
        self.sum = raw['sum']
        self.log_prod = raw['log_prod']

    def dump(self):
        return {
            'count': self.count,
            'sum': self.sum,
            'log_prod': self.log_prod,
        }


ctypedef double Sampler


cdef class Model_cy:
    cdef double alpha
    cdef double inv_beta

    def load(self, raw):
        self.alpha = raw['alpha']
        self.inv_beta = raw['inv_beta']

    def dump(self):
        return {
            'alpha': self.alpha,
            'inv_beta': self.inv_beta,
        }

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.count = 0
        group.sum = 0
        group.log_prod = 0.

    def group_add_value(self, Group group, int value):
        group.count += 1
        group.sum += value
        group.log_prod += log_factorial(value)

    def group_remove_value(self, Group group, int value):
        group.count -= 1
        group.sum -= value
        group.log_prod -= log_factorial(value)

    def group_merge(self, Group destin, Group source):
        destin.count += source.count
        destin.sum += source.sum
        destin.log_prod += source.log_prod

    cdef Model_cy plus_group(self, Group group):
        cdef Model_cy post = Model_cy()
        post.alpha = self.alpha + group.sum
        post.inv_beta = self.inv_beta + group.count
        return post

    #-------------------------------------------------------------------------
    # Sampling

    cpdef Sampler sampler_create(Model_cy self, Group group=None):
        cdef Model_cy post = self if group is None else self.plus_group(group)
        return sample_gamma(post.alpha, 1.0 / post.inv_beta)

    cpdef Value sampler_eval(self, Sampler sampler):
        return sample_poisson(sampler)

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
        cdef Model_cy post = self.plus_group(group)
        return gammaln(post.alpha + value) - gammaln(post.alpha) \
            + post.alpha * log(post.inv_beta) \
            - (post.alpha + value) * log(1. + post.inv_beta) \
            - log_factorial(value)

    def score_group(self, Group group):
        cdef Model_cy post = self.plus_group(group)
        return gammaln(post.alpha) - gammaln(self.alpha) \
            + self.alpha * log(self.inv_beta) \
            - post.alpha * log(post.inv_beta) \
            - group.log_prod

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alpha': 1., 'inv_beta': 1.},
            'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
        }
    ]


class GammaPoisson(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group


Model = GammaPoisson
