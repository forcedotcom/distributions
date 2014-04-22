from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions.lp.vector cimport VectorFloat

ctypedef float Value


cdef extern from "distributions/models/nich.hpp" namespace "distributions::normal_inverse_chi_sq":
    cppclass Model:
        float mu
        float kappa
        float sigmasq
        float nu
    cppclass Group:
        uint32_t count
        float mean
        float count_times_variance
        void init (Model &, rng_t &) nogil except +
        void add_value (Model &, Value &, rng_t &) nogil except +
        void remove_value (Model &, Value &, rng_t &) nogil except +
        void merge (Model &, Group &, rng_t &) nogil except +
    cppclass Sampler:
        float mu
        float sigma
        void init (Model &, Group &, rng_t &) nogil except +
        Value eval (Model &, rng_t &) nogil except +
    cppclass Scorer:
        float score
        float log_coeff
        float precision
        float mean
    cppclass Mixture:
        vector[Group] groups
        VectorFloat score
        VectorFloat log_coeff
        VectorFloat precision
        VectorFloat mean
        VectorFloat temp
        void init (Model &, rng_t &) nogil except +
        void add_group (Model &, rng_t &) nogil except +
        void remove_group (Model &, size_t) nogil except +
        void add_value \
            (Model &, size_t, Value &, rng_t &) nogil except +
        void remove_value \
            (Model &, size_t, Value &, rng_t &) nogil except +
        void score_value \
            (Model &, Value &, VectorFloat &, rng_t &) nogil except +


cdef extern from "distributions/models/nich.hpp" namespace "distributions":
    Value sample_value (Model &, Group &, rng_t &) nogil except +
    float score_value (Model &, Group &, Value &, rng_t &) nogil except +
    float score_group (Model &, Group &, rng_t &) nogil except +
