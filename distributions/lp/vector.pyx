from distributions.lp.vector cimport VectorFloat


cdef list vector_float_to_list(VectorFloat & vector):
    cdef list result = []
    cdef int i
    for i in xrange(vector.size()):
        result.append(vector.at(i))
    return result
