from libc.math cimport log, sqrt, lgamma


cdef double log_factorial(unsigned n):
    return lgamma(n + 1)


cdef double gammaln(double x):
    return lgamma(x)
