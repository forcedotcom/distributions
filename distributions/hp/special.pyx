from libc.math cimport log, sqrt, lgamma


cdef double logfactorial(unsigned n):
    return lgamma(n + 1)


cdef double gammaln(double x):
    return lgamma(x)
