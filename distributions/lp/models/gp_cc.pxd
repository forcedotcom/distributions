from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions.lp.vector cimport VectorFloat

ctypedef int Value


cdef extern from "distributions/models/gp.hpp" namespace "distributions::gamma_poisson":
    cppclass Model:
        float alpha
        float inv_beta
        Value sample_value (Group &, rng_t &) nogil except +
        float score_value (Group &, Value &, rng_t &) nogil except +
        float score_group (Group &, rng_t &) nogil except +
    cppclass Group:
        uint32_t count
        uint32_t sum
        float log_prod
        void init (Model &, rng_t &) nogil except +
        void add_value (Model &, Value &, rng_t &) nogil except +
        void remove_value (Model &, Value &, rng_t &) nogil except +
        void merge (Model &, Group &, rng_t &) nogil except +
    cppclass Sampler:
        float mean
        void init (Model &, Group &, rng_t &) nogil except +
        Value eval (Model &, rng_t &) nogil except +
    cppclass Scorer:
        float score
        float post_alpha
        float score_coeff
    cppclass Mixture:
        vector[Group] groups
        VectorFloat score
        VectorFloat post_alpha
        VectorFloat score_coeff
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
