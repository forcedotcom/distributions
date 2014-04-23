from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
cimport numpy
numpy.import_array()
from distributions.rng_cc cimport rng_t
from distributions.global_rng cimport get_rng
from distributions.lp.vector cimport (
    VectorFloat,
    vector_float_from_ndarray,
    vector_float_to_ndarray,
)
from distributions.mixins import GroupIoMixin, SharedIoMixin

cimport gp_cc as cc


cdef class __Shared:
    cdef cc.Model * ptr

    def __cinit__(self):
        self.ptr = new cc.Model()

    def __dealloc__(self):
        del self.ptr


cdef class __Group:
    cdef cc.Group * ptr

    def __cinit__(self):
        self.ptr = new cc.Group()

    def __dealloc__(self):
        del self.ptr

    def init(self, __Shared shared):
        self.ptr.init(shared.ptr[0], get_rng()[0])

    def add_value(self, __Shared shared, cc.Value value):
        self.ptr.add_value(shared.ptr[0], value, get_rng()[0])

    def remove_value(self, __Shared shared, cc.Value value):
        self.ptr.remove_value(shared.ptr[0], value, get_rng()[0])

    def merge(self, __Shared shared, __Group source):
        self.ptr.merge(shared.ptr[0], source.ptr[0], get_rng()[0])


cdef class __Mixture:
    cdef cc.Mixture * ptr
    cdef VectorFloat scores

    def __cinit__(self):
        self.ptr = new cc.Mixture()

    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.groups.size()

    def __getitem__(self, int groupid):
        assert groupid < len(self), "groupid out of bounds"
        group = __Group()
        group.ptr[0] = self.ptr.groups[groupid]
        return group

    def append(self, __Group group):
        self.ptr.groups.push_back(group.ptr[0])

    def clear(self):
        self.ptr.groups.clear()

    def init(self, __Shared shared):
        self.ptr.init(shared.ptr[0], get_rng()[0])

    def add_group(self, __Shared shared):
        self.ptr.add_group(shared.ptr[0], get_rng()[0])

    def remove_group(self, __Shared shared, int groupid):
        self.ptr.remove_group(shared.ptr[0], groupid)

    def add_value(self, __Shared shared, int groupid, cc.Value value):
        self.ptr.add_value(shared.ptr[0], groupid, value, get_rng()[0])

    def remove_value(self, __Shared shared, int groupid, cc.Value value):
        self.ptr.remove_value(shared.ptr[0], groupid, value, get_rng()[0])

    def score_value(self, __Shared shared, cc.Value value,
              numpy.ndarray[numpy.float32_t, ndim=1] scores_accum):
        assert len(scores_accum) == self.ptr.groups.size(), \
            "scores_accum != len(mixture)"
        vector_float_from_ndarray(self.scores, scores_accum)
        self.ptr.score_value(shared.ptr[0], value, self.scores, get_rng()[0])
        vector_float_to_ndarray(self.scores, scores_accum)


def __sample_value(__Shared shared, __Group group):
    cdef cc.Value value = cc.sample_value(
        shared.ptr[0], group.ptr[0], get_rng()[0])
    return value

def __sample_group(__Shared shared, int size):
    cdef __Group group = __Group()
    cdef cc.Sampler sampler
    sampler.init(shared.ptr[0], group.ptr[0], get_rng()[0])
    cdef list result = []
    cdef int i
    cdef cc.Value value
    for i in xrange(size):
        value = sampler.eval(shared.ptr[0], get_rng()[0])
        result.append(value)
    return result

def __score_value(__Shared shared, __Group group, cc.Value value):
    return cc.score_value(shared.ptr[0], group.ptr[0], value, get_rng()[0])

def __score_group(__Shared shared, __Group group):
    return cc.score_group(shared.ptr[0], group.ptr[0], get_rng()[0])


#############################################################################


NAME = 'GammaPoisson'
EXAMPLES = [
    {
        'shared': {'alpha': 1., 'inv_beta': 1.},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    },
]
Value = int


cdef class _Shared(__Shared):
    def load(self, raw):
        self.ptr.alpha = raw['alpha']
        self.ptr.inv_beta = raw['inv_beta']

    def dump(self):
        return {
            'alpha': self.ptr.alpha,
            'inv_beta': self.ptr.inv_beta,
        }


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(__Group):
    def load(self, raw):
        self.ptr.count = raw['count']
        self.ptr.sum = raw['sum']
        self.ptr.log_prod = raw['log_prod']

    def dump(self):
        return {
            'count': self.ptr.count,
            'sum': self.ptr.sum,
            'log_prod': self.ptr.log_prod,
        }


class Group(_Group, GroupIoMixin):
    pass


Mixture = __Mixture
sample_value = __sample_value
sample_group = __sample_group
score_value = __score_value
score_group = __score_group
