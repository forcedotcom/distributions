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
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin


NAME = 'GammaPoisson'
EXAMPLES = [
    {
        'shared': {'alpha': 1., 'inv_beta': 1.},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    }
]
Value = int


ctypedef int _Value


cdef class _Shared:
    cdef double alpha
    cdef double inv_beta

    cdef _Shared plus_group(self, _Group group):
        cdef _Shared post = self.__class__()
        post.alpha = self.alpha + group.sum
        post.inv_beta = self.inv_beta + group.count
        return post

    def load(self, raw):
        self.alpha = raw['alpha']
        self.inv_beta = raw['inv_beta']

    def dump(self):
        return {
            'alpha': self.alpha,
            'inv_beta': self.inv_beta,
        }


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group:
    cdef int count
    cdef int sum
    cdef double log_prod

    def init(self, _Shared shared):
        self.count = 0
        self.sum = 0
        self.log_prod = 0.

    def add_value(self, _Shared shared, int value):
        self.count += 1
        self.sum += value
        self.log_prod += log_factorial(value)

    def add_repeated_value(self, _Shared shared, int value, int count):
        self.count += count
        self.sum += count * value
        self.log_prod += count * log_factorial(value)

    def remove_value(self, _Shared shared, int value):
        self.count -= 1
        self.sum -= value
        self.log_prod -= log_factorial(value)

    def merge(self, _Shared shared, _Group source):
        self.count += source.count
        self.sum += source.sum
        self.log_prod += source.log_prod

    def score_value(self, _Shared shared, _Value value):
        cdef _Shared post = shared.plus_group(self)
        return gammaln(post.alpha + value) - gammaln(post.alpha) \
            + post.alpha * log(post.inv_beta) \
            - (post.alpha + value) * log(1. + post.inv_beta) \
            - log_factorial(value)

    def score_data(self, _Shared shared):
        cdef _Shared post = shared.plus_group(self)
        return gammaln(post.alpha) - gammaln(shared.alpha) \
            + shared.alpha * log(shared.inv_beta) \
            - post.alpha * log(post.inv_beta) \
            - self.log_prod

    def sample_value(self, _Shared shared):
        cdef Sampler sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

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


class Group(_Group, GroupIoMixin):
    pass


cdef class Sampler:
    cdef double lambda_

    def init(self, _Shared shared, _Group group=None):
        cdef _Shared post
        post = shared if group is None else shared.plus_group(group)
        self.lambda_ = sample_gamma(post.alpha, 1.0 / post.inv_beta)

    def eval(self, _Shared shared):
        return sample_poisson(self.lambda_)


def sample_group(_Shared shared, int size):
    cdef Sampler sampler = Sampler()
    sampler.init(shared)
    cdef list result = []
    cdef int i
    for i in xrange(size):
        result.append(sampler.eval(shared))
    return result
