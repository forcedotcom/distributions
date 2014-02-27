import numpy
cimport numpy


cdef extern from "distributions/random.hpp" namespace "distributions":
    cppclass rng_t
    cdef rng_t global_rng


cdef extern from 'distributions/std_wrapper.hpp' namespace 'std_wrapper':
    cdef void std_rng_seed(unsigned long seed)
    cdef double std_random_normal(double mu, double sigmasq)
    cdef double std_random_chisq(double nu)
    cdef double std_random_gamma(double alpha, double beta)
    cdef int std_random_poisson(double mu)
    cdef int std_random_categorical(size_t dim, double ps[])
    cdef void std_random_dirichlet(
            size_t dim,
            double alphas[],
            double thetas[])


cpdef seed(unsigned long s):
    std_rng_seed(s)


cpdef double sample_normal(double mu, double sigmasq):
    return std_random_normal(mu, sigmasq)


cpdef double sample_chisq(double nu):
    return std_random_chisq(nu)


cpdef sample_gamma(double a, double b):
    return std_random_gamma(a, b)


cpdef sample_poisson(double mu):
    return std_random_poisson(mu)


cdef sample_discrete(size_t dim, double ps[]):
    return std_random_categorical(dim, ps)


cdef sample_dirichlet(size_t dim, double alphas[], double thetas[]):
    std_random_dirichlet(dim, alphas, thetas)


#def sample_discrete(numpy.ndarray[double, ndim=1] ps):
#    cdef int dim = ps.shape[0]
#    return std_random_categorical(dim, &ps[0])
#
#
#def sample_dirichlet(
#        numpy.ndarray[double, ndim=1] alphas,
#        numpy.ndarray[double, ndim=1] thetas):
#    cdef int dim = alphas.shape[0]
#    std_random_dirichlet(dim, &alphas[0], &thetas[0])
