from libcpp.utility cimport pair

cdef extern from "distributions/sparse_counter.hpp":
    cdef cppclass SparseCounter "distributions::SparseCounter<uint32_t, uint32_t>":
        cppclass iterator:
            pair[int, int]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        SparseCounter() nogil except +
        SparseCounter(SparseCounter&) nogil except +
        void clear() nogil
        void init_count(int, int) nogil
        int get_count(int) nogil
        int get_total() nogil
        void add(int) nogil
        void remove(int) nogil
        void merge(SparseCounter&) nogil
        #SparseCounter& operator=(SparseCounter&)
        iterator begin() nogil
        iterator end() nogil
        #bint empty() nogil
        #size_t size() nogil
        #void swap(SparseCounter&) nogil
