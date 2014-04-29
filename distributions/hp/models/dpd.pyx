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
from cython.operator cimport dereference as deref, preincrement as inc
from distributions.hp.special cimport log, gammaln
from distributions.hp.random cimport sample_dirichlet, sample_discrete
from distributions.sparse_counter cimport SparseCounter
from distributions.mixins import GroupIoMixin, SharedIoMixin


NAME = 'DirichletProcessDiscrete'
EXAMPLES = [
    {
        'shared': {
            'gamma': 0.5,
            'alpha': 0.5,
            'betas': {  # beta0 must be zero for unit tests
                '0': 0.25,
                '1': 0.5,
                '2': 0.25,
            },
        },
        'values': [0, 1, 0, 2, 0, 1, 0],
    },
]
Value = int


ctypedef int _Value
cpdef _Value OTHER = -1


cdef class _Shared:
    cdef double gamma
    cdef double alpha
    cdef double beta0
    #cdef numpy.ndarray[double, ndim=1] betas
    cdef numpy.ndarray betas

    def load(self, dict raw):
        self.gamma = raw['gamma']
        self.alpha = raw['alpha']
        cdef dict raw_betas = raw['betas']
        cdef int i
        cdef list betas = [raw_betas[str(i)] for i in xrange(len(raw_betas))]
        self.betas = numpy.array(betas, dtype=numpy.float)  # dense
        self.betas0 = 1.0 - self.betas.sum()

    def dump(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'betas': {str(i): beta for i, beta in enumerate(self.betas)},
        }


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group:
    cdef SparseCounter * counts

    def __cinit__(self):
        self.counts = new SparseCounter()

    def __dealloc__(self):
        del self.counts

    def init(self, _Shared shared):
        self.counts.clear()

    def add_value(self, _Shared shared, _Value value):
        assert value != OTHER, 'tried to add OTHER to group'
        self.counts.add(value)

    def remove_value(self, _Shared shared, _Value value):
        assert value != OTHER, 'tried to remove OTHER to group'
        self.counts.remove(value)

    def merge(self, _Shared shared, _Group source):
        self.counts.merge(source.counts[0])

    def load(self, dict raw):
        cdef SparseCounter * counts = self.counts
        counts.clear()
        cdef dict raw_counts = raw['counts']
        cdef str i
        cdef int count
        for i, count in raw_counts.iteritems():
            self.counts.init_count(int(i), count)

    def dump(self):
        cdef dict counts = {}
        cdef SparseCounter.iterator it = self.counts.begin()
        cdef SparseCounter.iterator end = self.counts.end()
        while it != end:
            counts[str(deref(it).first)] = deref(it).second
            inc(it)
        return {'counts': counts}


class Group(_Group, GroupIoMixin):
    pass


# Buffer types only allowed as function local variables
#ctypedef numpy.ndarray[numpy.float64_t, ndim=1] Sampler
ctypedef numpy.ndarray _Sampler


def sampler_create(_Shared shared, _Group group=None):
    cdef int size = shared.betas.shape[0]
    cdef numpy.ndarray[double, ndim=1] probs = \
        numpy.zeros(size + 1, dtype=numpy.double)
    probs[:-1] = shared.betas * shared.alpha
    probs[-1] = shared.beta0 * shared.alpha
    cdef SparseCounter.iterator it
    cdef SparseCounter.iterator end
    if group is not None:
        counts = group.counts
        it = group.counts.begin()
        end = group.counts.end()
        while it != end:
            probs[deref(it).first] += deref(it).second
            inc(it)
    cdef numpy.ndarray[double, ndim=1] sampler = \
        numpy.zeros(size + 1, dtype=numpy.double)
    sample_dirichlet(
        size + 1,
        <double *> probs.data,
        <double *> sampler.data)
    return sampler

def sampler_eval(_Shared shared, _Sampler sampler):
    cdef int size = len(sampler)
    cdef int index = sample_discrete(size + 1, <double *> sampler.data)
    if index == size:
        return OTHER
    else:
        return index

def sample_value(_Shared shared, _Group group):
    cdef _Sampler sampler = sampler_create(shared, group)
    return sampler_eval(shared, sampler)

def sample_group(_Shared shared, int size):
    cdef _Sampler sampler = sampler_create(shared)
    cdef list result = []
    cdef int i
    for i in xrange(size):
        result.append(sampler_eval(shared, sampler))
    return result

def score_value(_Shared shared, _Group group, _Value value):
    cdef SparseCounter * counts = group.counts
    cdef double denom = shared.alpha + counts.get_total()
    cdef double numer
    if value == OTHER:
        numer = shared.beta0 * shared.alpha
    else:
        numer = shared.betas[value] * shared.alpha + counts.get_count(value)
    return log(numer / denom)

def score_group(_Shared shared, _Group group):
    assert len(shared.betas), 'betas is empty'
    cdef double score = 0.
    cdef SparseCounter * counts = group.counts
    cdef SparseCounter.iterator it = counts.begin()
    cdef SparseCounter.iterator end = counts.end()
    cdef int i
    cdef int count
    cdef double prior_i
    while it != end:
        i = deref(it).first
        count = deref(it).second
        prior_i = shared.betas[i] * shared.alpha
        score += gammaln(prior_i + count) - gammaln(prior_i)
        inc(it)
    score += gammaln(shared.alpha) - gammaln(shared.alpha + counts.get_total())

    return score
