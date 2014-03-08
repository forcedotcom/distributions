from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from 'distributions/clustering.hpp':
    cppclass rng_t

    cppclass Assignments_cc "distributions::Clustering<int>::Assignments":
        Assignments_cc() nogil except +
        Assignments_cc(Assignments_cc &) nogil except +
        cppclass iterator:
            pair[int, int] & operator*() nogil
            iterator operator++() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        int & operator[](int) nogil
        iterator begin() nogil
        iterator end() nogil
        void clear() nogil
        bint empty() nogil

    cdef vector[int] count_assignments \
            "distributions::Clustering<int>::count_assignments" \
            (Assignments_cc & assignments) nogil

    cppclass PitmanYor_cc "distributions::Clustering<int>::PitmanYor":
        vector[int] sample_assignments(int size, rng_t & rng) nogil
        float score_counts(vector[int] & counts) nogil
        float score_add_value (
                int this_group_size,
                int total_group_count,
                int total_value_count) nogil
        float score_remove_value (
                int this_group_size,
                int total_group_count,
                int total_value_count) nogil


cdef class Assignments:
    cdef Assignments_cc * ptr
    def __cinit__(self):
        self.ptr = new Assignments_cc()
    def __dealloc__(self):
        del self.ptr

    def load(self, raw):
        self.ptr.clear()
        cdef int value_id
        cdef int group_id
        for value_id, group_id in raw.iteritems():
            deref(self.ptr)[value_id] = group_id

    def dump(self):
        cdef dict raw = {}
        cdef Assignments_cc.iterator i = self.ptr.begin()
        while i != self.ptr.end():
            deref(self.ptr)[deref(i).first] = deref(i).second
            inc(i)
        return raw


cdef class PitmanYor:
    cdef PitmanYor_cc * ptr
    def __cinit__(self):
        self.ptr = new PitmanYor_cc()
    def __dealloc__(self):
        del self.ptr
