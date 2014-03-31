import distributions.rng
cimport distributions.rng_cc


cdef rng_t * get_rng():
    return distributions.rng_cc.extract_rng(distributions.rng.global_rng.cc)
