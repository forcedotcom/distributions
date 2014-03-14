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
from distributions.mixins import ComponentModel, Serializable

cpdef int MAX_DIM = 256

#-------------------------------------------------------------------------
# Datatypes

ctypedef int Value


cdef class Group:
    cdef int counts[256]
    cdef int dim  # only required for dumping
    def __cinit__(self):
        self.dim = 0

    def load(self, raw):
        counts = raw['counts']
        self.dim = len(counts)
        assert self.dim <= MAX_DIM
        cdef int i
        for i in xrange(self.dim):
            self.counts[i] = counts[i]

    def dump(self):
        return {'counts': [self.counts[i] for i in xrange(self.dim)]}


# Buffer types only allowed as function local variables
#ctypedef numpy.ndarray[numpy.float64_t, ndim=1] Sampler
ctypedef numpy.ndarray Sampler


cdef class Model_cy:
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

    #-------------------------------------------------------------------------
    # Mutation

    def group_init(self, Group group):
        group.dim = self.dim
        cdef int i
        for i in xrange(self.dim):
            group.counts[i] = 0

    def group_add_value(self, Group group, int value):
        group.counts[value] += 1

    def group_remove_value(self, Group group, int value):
        group.counts[value] -= 1

    def group_merge(self, Group destin, Group source):
        cdef int i
        for i in xrange(self.dim):
            destin.counts[i] += source.counts[i]

    #-------------------------------------------------------------------------
    # Sampling

    cpdef Sampler sampler_create(self, Group group=None):
        cdef Sampler sampler = numpy.zeros(self.dim, dtype=numpy.float64)
        cdef double * ps = <double *> sampler.data
        cdef int i
        if group is None:
            for i in xrange(self.dim):
                sampler[i] = self.alphas[i]
        else:
            for i in xrange(self.dim):
                sampler[i] = group.counts[i] + self.alphas[i]
        sample_dirichlet(self.dim, ps, ps)
        return sampler

    cpdef Value sampler_eval(self, Sampler sampler):
        cdef double * ps = <double *> sampler.data
        return sample_discrete(self.dim, ps)

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

    def score_value(self, Group group, Value value):
        """
        McCallum, et. al, 'Rethinking LDA: Why Priors Matter' eqn 4
        """
        cdef double total = 0.0
        cdef int i
        for i in xrange(self.dim):
            total += group.counts[i] + self.alphas[i]
        return log((group.counts[value] + self.alphas[value]) / total)

    def score_group(self, Group group):
        """
        From equation 22 of Michael Jordan's CS281B/Stat241B
        Advanced Topics in Learning and Decision Making course,
        'More on Marginal Likelihood'
        """
        cdef int i
        cdef double alpha_sum = 0.0
        cdef int count_sum = 0
        cdef double sum = 0.0
        for i in xrange(self.dim):
            alpha_sum += self.alphas[i]
        for i in xrange(self.dim):
            count_sum += group.counts[i]
        for i in xrange(self.dim):
            sum += (gammaln(self.alphas[i] + group.counts[i])
                    - gammaln(self.alphas[i]))
        return sum + gammaln(alpha_sum) - gammaln(alpha_sum + count_sum)

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {
            'model': {'alphas': [1.0, 4.0]},
            'values': [0, 1, 1, 1, 1, 0, 1],
        },
        {
            'model': {'alphas': [0.5, 0.5, 0.5, 0.5]},
            'values': [0, 1, 0, 2, 0, 1, 0],
        },
    ]


class DirichletDiscrete(Model_cy, ComponentModel, Serializable):

    #-------------------------------------------------------------------------
    # Datatypes

    Value = int

    Group = Group


Model = DirichletDiscrete
