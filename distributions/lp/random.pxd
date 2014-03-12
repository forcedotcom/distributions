cdef extern from 'distributions/std_wrapper.hpp' namespace 'std_wrapper':
    cppclass rng_t:
        rng_t() nogil except +
        rng_t(int) nogil except +
        rng_t(rng_t & rng) nogil except +
        int sample "operator()" () nogil
        void seed(int) nogil
    cdef rng_t global_rng
