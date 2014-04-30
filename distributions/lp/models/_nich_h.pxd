from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions.lp.vector cimport VectorFloat
from distributions.sparse_counter cimport SparseCounter


ctypedef float Value


cdef extern from "distributions/models/nich.hpp" namespace "distributions::normal_inverse_chi_sq":
    cppclass Shared:
        float mu
        float kappa
        float sigmasq
        float nu


    cppclass Group:
        uint32_t count
        float mean
        float count_times_variance
        void init (Shared &, rng_t &) nogil except +
        void add_value (Shared &, Value &, rng_t &) nogil except +
        void remove_value (Shared &, Value &, rng_t &) nogil except +
        void merge (Shared &, Group &, rng_t &) nogil except +



    cppclass Sampler:
        void init (Shared &, Group &, rng_t &) nogil except +
        Value eval (Shared &, rng_t &) nogil except +


    cppclass Mixture:
        vector[Group] groups "groups()"
        void init (Shared &, rng_t &) nogil except +
        void add_group (Shared &, rng_t &) nogil except +
        void remove_group (Shared &, size_t) nogil except +
        void add_value \
            (Shared &, size_t, Value &, rng_t &) nogil except +
        void remove_value \
            (Shared &, size_t, Value &, rng_t &) nogil except +
        void score_value \
            (Shared &, Value &, VectorFloat &, rng_t &) nogil except +



    Value sample_value (Shared &, Group &, rng_t &) nogil except +
    float score_value (Shared &, Group &, Value &, rng_t &) nogil except +
    float score_group (Shared &, Group &, rng_t &) nogil except +
