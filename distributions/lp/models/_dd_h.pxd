from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions.lp.vector cimport VectorFloat
from distributions.sparse_counter cimport SparseCounter


ctypedef int Value


cdef extern from "distributions/models/dd.hpp" namespace "distributions::dirichlet_discrete":
    cppclass Shared "distributions::dirichlet_discrete::Shared<256>":
        int dim
        float alphas[256]


    cppclass Group "distributions::dirichlet_discrete::Group<256>":
        uint32_t count_sum
        uint32_t counts[]
        void init (Shared &, rng_t &) nogil except +
        void add_value (Shared &, Value &, rng_t &) nogil except +
        void remove_value (Shared &, Value &, rng_t &) nogil except +
        void merge (Shared &, Group &, rng_t &) nogil except +
        float score_value (Shared &, Value &, rng_t &) nogil except +
        float score_data (Shared &, rng_t &) nogil except +


    cppclass Sampler "distributions::dirichlet_discrete::Sampler<256>":
        void init (Shared &, Group &, rng_t &) nogil except +
        Value eval (Shared &, rng_t &) nogil except +


    cppclass Mixture "distributions::dirichlet_discrete::Mixture<256>":
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
        float score_data (Shared &, rng_t &) nogil except +


    Value sample_value (Shared &, Group &, rng_t &) nogil except +
