import distributions.rng


cdef class RngCc:
    def __cinit__(self):
        self.ptr = new rng_t()
    def __dealloc__(self):
        del self.ptr

cdef rng_t * extract_rng(RngCc rng):
    return rng.ptr

cdef rng_t * get_rng():
    return extract_rng(distributions.rng.global_rng.cc)
