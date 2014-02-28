from libcpp.utility cimport pair

cdef extern from "sparse_counter.hpp":
    cdef cppclass SparseCounter_iterator "SparseCounter<uint32_t, uint32_t>::iterator":
        pair[int, int]& operator*() nogil
        SparseCounter_iterator operator++() nogil
        SparseCounter_iterator operator--() nogil
        bint operator==(SparseCounter_iterator) nogil
        bint operator!=(SparseCounter_iterator) nogil
    cdef cppclass SparseCounter "SparseCounter<uint32_t, uint32_t>":
        SparseCounter() nogil except +
        SparseCounter(SparseCounter&) nogil except +
        void clear() nogil
        void init_count(int, int) nogil
        int get_count(int) nogil
        int get_total() nogil
        void add(int) nogil
        void remove(int) nogil
        #SparseCounter& operator=(SparseCounter&)
        SparseCounter_iterator begin() nogil
        SparseCounter_iterator end() nogil
        #bint empty() nogil
        #size_t size() nogil
        #void swap(SparseCounter&) nogil
