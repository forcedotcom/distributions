cdef extern from "distributions/vector.hpp" namespace "distributions":
    cppclass VectorFloat:
        VectorFloat() nogil except +
        size_t size () nogil
        float & at "operator[]" (size_t index) nogil


cdef extern from "distributions/vector_math.hpp" namespace "distributions":
    cdef vector_add(size_t size, float * io, float * in1) nogil


cdef list vector_float_to_list(VectorFloat & vector)
