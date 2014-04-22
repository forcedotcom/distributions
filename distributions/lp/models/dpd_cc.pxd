from libcpp.vector cimport vector

from distributions.rng_cc cimport rng_t
from distributions.lp.vector cimport VectorFloat
from distributions.sparse_counter cimport SparseCounter

cimport dpd_cc as cc
ctypedef unsigned Value

cdef extern from "distributions/models/dpd.hpp" namespace "distributions::dirichlet_process_discrete":
    cppclass Model:
        float gamma
        float alpha
        float beta0
        vector[float] betas
    cppclass Group:
        SparseCounter counts
        void init (Model &, rng_t &) nogil except +
        void add_value (Model &, Value &, rng_t &) nogil except +
        void remove_value (Model &, Value &, rng_t &) nogil except +
        void merge (Model &, Group &, rng_t &) nogil except +
    cppclass Sampler:
        vector[float] probs
        void init (Model &, Group &, rng_t &) nogil except +
        Value eval (Model &, rng_t &) nogil except +
    cppclass Scorer:
        vector[float] scores
    cppclass Mixture:
        vector[Group] groups
        vector[VectorFloat] scores
        VectorFloat scores_shift
        void init (Model &, rng_t &) nogil except +
        void add_group (Model &, rng_t &) nogil except +
        void remove_group (Model &, size_t) nogil except +
        void add_value \
            (Model &, size_t, Value &, rng_t &) nogil except +
        void remove_value \
            (Model &, size_t, Value &, rng_t &) nogil except +
        void score_value \
            (Model &, Value &, VectorFloat &, rng_t &) nogil except +


cdef extern from "distributions/models/dpd.hpp" namespace "distributions":
    Value sample_value (Model &, Group &, rng_t &) nogil except +
    float score_value (Model &, Group &, Value &, rng_t &) nogil except +
    float score_group (Model &, Group &, rng_t &) nogil except +
