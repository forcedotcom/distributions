cpdef int MAX_DIM = 256

cdef extern from "common.hpp" namespace "distributions":
    cppclass rng_t:
        pass

cdef extern from "models/dd.hpp" namespace "distributions:DirichletDiscrete<256>":

    ctypedef int value_t

    cppclass model_t:
        int dim
        float alphas[]

    cppclass group_t:
        int counts[]

    cpdef void group_init_prior(
            group_t group,
            model_t model,
            rng_t rng)

    cpdef void group_add_data(
            value_t value,
            group_t group,
            model_t model)

    cpdef void group_rem_data(
            value_t value,
            group_t group,
            model_t model)

    cpdef float score_add(
            value_t value,
            group_t group,
            model_t model,
            rng_t rng)

    cpdef float score_group(
            group_t group,
            model_t model,
            rng_t rng)


cdef extern from "common.hpp" namespace "distributions":
    int foo()


def wrapped_foo():
    return foo() + 2
