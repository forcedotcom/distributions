cdef extern from 'rng_cc.hpp' namespace 'std_wrapper':
    cdef cppclass rng_t:
        rng_t() nogil except +
        rng_t(int) nogil except +
        rng_t(rng_t & rng) nogil except +
        int sample "operator()" () nogil
        void seed(int) nogil

cdef class RngCc:
    cdef rng_t * ptr

cdef rng_t * get_rng()
