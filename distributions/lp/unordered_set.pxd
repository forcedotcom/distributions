from libcpp.pair cimport pair


cdef extern from "<unordered_set>" namespace "std":
    cdef cppclass unordered_set[T]:
        cppclass iterator:
            T& operator*()
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        cppclass reverse_iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(reverse_iterator) nogil
            bint operator!=(reverse_iterator) nogil
        unordered_set() nogil except +
        unordered_set(unordered_set&) nogil except +
        bint operator==(unordered_set&, unordered_set&) nogil
        bint operator!=(unordered_set&, unordered_set&) nogil
        bint operator<(unordered_set&, unordered_set&) nogil
        bint operator>(unordered_set&, unordered_set&) nogil
        bint operator<=(unordered_set&, unordered_set&) nogil
        bint operator>=(unordered_set&, unordered_set&) nogil
        iterator begin() nogil
        void clear() nogil
        size_t count(T&) nogil
        bint empty() nogil
        iterator end() nogil
        pair[iterator, iterator] equal_range(T&) nogil
        void erase(iterator) nogil
        void erase(iterator, iterator) nogil
        size_t erase(T&) nogil
        iterator find(T&) nogil
        pair[iterator, bint] insert(T&) nogil
        iterator insert(iterator, T&) nogil
        iterator lower_bound(T&) nogil
        size_t max_size() nogil
        reverse_iterator rbegin() nogil
        reverse_iterator rend() nogil
        size_t size() nogil
        void swap(unordered_set&) nogil
        iterator upper_bound(T&) nogil
