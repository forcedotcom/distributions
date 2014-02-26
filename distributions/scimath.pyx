from libc.math cimport sqrt, lgamma


cdef extern from 'std_wrapper.hpp' namespace 'std_wrapper':
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


cpdef double normal_draw(double mu, double sigmasq):
    return std_random_normal(mu, sigmasq)


cpdef double chisq_draw(double nu):
    return std_random_chisq(nu)


cpdef gamma_draw(double a, double b):
    return std_random_gamma(a, b)


cpdef poisson_draw(double mu):
    return std_random_poisson(mu)


cdef categorical_draw(size_t K, double p[]):
    return std_random_categorical(K, p)


cdef dirichlet_draw(size_t K, double alpha[], double theta[]):
    std_random_dirichlet(K, alpha, theta)



cdef double logfactorial(unsigned n):
    return lgamma(n + 1)


cdef double gammaln(double x):
    return lgamma(x)
