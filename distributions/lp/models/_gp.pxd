cimport _gp_h as _h


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


cdef class Shared:
    cdef _h.Shared * ptr


cdef class Group:
    cdef _h.Group * ptr


cdef class Mixture:
    cdef _h.Mixture * ptr
    cdef VectorFloat scores
