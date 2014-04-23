from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions.lp.vector cimport VectorFloat
from distributions.sparse_counter cimport SparseCounter


ctypedef int Value


cdef extern from "distributions/models/dd.hpp" namespace "distributions::dirichlet_discrete":
    cppclass Model "distributions::dirichlet_discrete::Model<256>":
        int dim
        float alphas[256]


    cppclass Group "distributions::dirichlet_discrete::Group<256>":
        uint32_t count_sum
        uint32_t counts[]
        void init (Model &, rng_t &) nogil except +
        void add_value (Model &, Value &, rng_t &) nogil except +
        void remove_value (Model &, Value &, rng_t &) nogil except +
        void merge (Model &, Group &, rng_t &) nogil except +



    cppclass Sampler "distributions::dirichlet_discrete::Sampler<256>":
        void init (Model &, Group &, rng_t &) nogil except +
        Value eval (Model &, rng_t &) nogil except +


    cppclass Mixture "distributions::dirichlet_discrete::Mixture<256>":
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



cdef extern from "distributions/models/dd.hpp" namespace "distributions":
    Value sample_value (Model &, Group &, rng_t &) nogil except +
    float score_value (Model &, Group &, Value &, rng_t &) nogil except +
    float score_group (Model &, Group &, rng_t &) nogil except +
