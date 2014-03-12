from libcpp.vector cimport vector


cdef extern from "distributions/special.hpp":
    cdef float _fast_log "distributions::fast_log" (float x)
    cdef float _fast_lgamma "distributions::fast_lgamma" (float y)
    cdef float _fast_lgamma_nu "distributions::fast_lgamma_nu" (float nu)
    cdef vector[float] _log_stirling1_row \
            "distributions::log_stirling1_row" (int n)


cpdef float fast_log(float x):
    return _fast_log(x)


cpdef float fast_lgamma(float y):
    return _fast_lgamma(y)


cpdef float fast_lgamma_nu(float nu):
    return _fast_lgamma_nu(nu)


cpdef list log_stirling1_row(int n):
    return _log_stirling1_row(n)
