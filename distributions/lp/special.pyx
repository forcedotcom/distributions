from libcpp.vector cimport vector


cdef extern from "distributions/special.hpp":
    cdef vector[float] _log_stirling1_row \
            "distributions::log_stirling1_row" (int n)


cpdef log_stirling1_row(int n):
    cdef list row = _log_stirling1_row(n)
    return row
