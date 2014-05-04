cimport _dpd
import _dpd

from cython.operator cimport dereference as deref, preincrement as inc
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


cdef class _Shared(_dpd.Shared):
    def load(self, dict raw):
        self.ptr.gamma = raw['gamma']
        self.ptr.alpha = raw['alpha']
        self.ptr.betas.clear()
        cdef dict raw_betas = raw['betas']
        self.ptr.betas.resize(len(raw_betas))
        cdef str i
        cdef float beta
        cdef double beta0 = 1.0
        for i, beta in raw_betas.iteritems():
            self.ptr.betas[int(i)] = beta
            beta0 -= beta
        self.ptr.beta0 = beta0

    def dump(self):
        cdef dict betas = {}
        cdef int i
        for i in xrange(self.ptr.betas.size()):
            betas[str(i)] = self.ptr.betas[i]
        return {
            'gamma': float(self.ptr.gamma),
            'alpha': float(self.ptr.alpha),
            'betas': betas,
        }


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(_dpd.Group):
    def load(self, dict raw):
        cdef SparseCounter * counts = & self.ptr.counts
        counts.clear()
        cdef dict raw_counts = raw['counts']
        cdef str i
        cdef int count
        for i, count in raw_counts.iteritems():
            counts.init_count(int(i), count)

    def dump(self):
        cdef dict counts = {}
        cdef SparseCounter.iterator it = self.ptr.counts.begin()
        cdef SparseCounter.iterator end = self.ptr.counts.end()
        while it != end:
            counts[str(deref(it).first)] = deref(it).second
            inc(it)
        return {'counts': counts}


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_dpd.Sampler):
    pass


Mixture = _dpd.Mixture
sample_value = _dpd.sample_value
sample_group = _dpd.sample_group
score_group = _dpd.score_group
