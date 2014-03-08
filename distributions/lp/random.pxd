cdef extern from 'distributions/std_wrapper.hpp' namespace 'std_wrapper':
    cppclass rng_t
    cdef rng_t global_rng
