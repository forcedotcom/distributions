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
from distributions.hp.special cimport log, gammaln
from distributions.hp.random cimport sample_dirichlet, sample_discrete
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin

cpdef int MAX_DIM = 256

NAME = 'DirichletDiscrete'
EXAMPLES = [
    {
        'shared': {'alphas': [0.5, 0.5, 0.5, 0.5]},
        'values': [0, 1, 0, 2, 0, 1, 0],
    },
    {
        'shared': {'alphas': [1.0, 4.0]},
        'values': [0, 1, 1, 1, 1, 0, 1],
    },
    {
        'shared': {'alphas': [2.0 / n for n in xrange(1, 21)]},
        'values': range(20),
    },
]
Value = int


ctypedef int _Value


cdef class _Shared:
    cdef double[256] alphas
    cdef int dim

    def __cinit__(self):
        self.dim = 0

    def load(self, raw):
        alphas = raw['alphas']
        self.dim = len(alphas)
        assert self.dim <= MAX_DIM
        cdef int i
        for i in xrange(self.dim):
            self.alphas[i] = alphas[i]

    def dump(self):
        return {'alphas': [self.alphas[i] for i in xrange(self.dim)]}


class Shared(_Shared, SharedMixin, SharedIoMixin):
    pass


cdef class _Group:
    cdef int counts[256]
    cdef int dim  # only required for dumping

    def __cinit__(self):
        self.dim = 0

    def init(self, _Shared shared):
        self.dim = shared.dim
        cdef int i
        for i in xrange(self.dim):
            self.counts[i] = 0

    def add_value(self, _Shared shared, int value):
        self.counts[value] += 1

    def add_repeated_value(self, _Shared shared, int value, int count):
        self.counts[value] += count

    def remove_value(self, _Shared shared, int value):
        self.counts[value] -= 1

    def merge(self, _Shared shared, _Group source):
        cdef int i
        for i in xrange(self.dim):
            self.counts[i] += source.counts[i]

    def score_value(self, _Shared shared, _Value value):
        """
        McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
        """
        cdef double total = 0.0
        cdef int i
        for i in xrange(shared.dim):
            total += self.counts[i] + shared.alphas[i]
        return log((self.counts[value] + shared.alphas[value]) / total)

    def score_data(self, _Shared shared):
        """
        From equation 22 of Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """
        cdef int i
        cdef double alpha_sum = 0.0
        cdef int count_sum = 0
        cdef double sum = 0.0
        for i in xrange(shared.dim):
            alpha_sum += shared.alphas[i]
        for i in xrange(shared.dim):
            count_sum += self.counts[i]
        for i in xrange(shared.dim):
            sum += (gammaln(shared.alphas[i] + self.counts[i])
                    - gammaln(shared.alphas[i]))
        return sum + gammaln(alpha_sum) - gammaln(alpha_sum + count_sum)

    def sample_value(self, _Shared shared):
        cdef Sampler sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, raw):
        counts = raw['counts']
        self.dim = len(counts)
        assert self.dim <= MAX_DIM
        cdef int i
        for i in xrange(self.dim):
            self.counts[i] = counts[i]

    def dump(self):
        return {'counts': [self.counts[i] for i in xrange(self.dim)]}


class Group(_Group, GroupIoMixin):
    pass


cdef class Sampler:
    # Buffer types only allowed as function local variables
    #cdef numpy.ndarray[numpy.float64_t, ndim=1] ps
    cdef numpy.ndarray ps

    def init(self, _Shared shared, _Group group=None):
        self.ps = numpy.zeros(shared.dim, dtype=numpy.float64)
        cdef double * ps = <double *> self.ps.data
        cdef int i
        if group is None:
            for i in xrange(shared.dim):
                ps[i] = shared.alphas[i]
        else:
            for i in xrange(shared.dim):
                ps[i] = group.counts[i] + shared.alphas[i]
        sample_dirichlet(shared.dim, ps, ps)

    def eval(self, _Shared shared):
        cdef double * ps = <double *> self.ps.data
        return sample_discrete(shared.dim, ps)

def sample_group(_Shared shared, int size):
    cdef Sampler sampler = Sampler()
    sampler.init(shared)
    cdef list result = []
    cdef int i
    for i in xrange(size):
        result.append(sampler.eval(shared))
    return result
