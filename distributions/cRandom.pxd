cdef extern from "distributions/std_wrapper.hpp" namespace "std_wrapper":
    cppclass rng_t
    cdef rng_t global_rng


cpdef int random()
cpdef seed(unsigned long s)
cpdef double sample_normal(double mu, double sigmasq)
cpdef double sample_chisq(double nu)
cpdef sample_gamma(double a, double b)
cpdef sample_poisson(double mu)
cdef sample_discrete(size_t dim, double ps[])
cdef sample_dirichlet(size_t dim, double alphas[], double thetas[])
