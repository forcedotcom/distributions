# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cimport numpy


cdef extern from "distributions/vector.hpp" namespace "distributions":
    cdef cppclass VectorFloat:
        cppclass iterator:
            float& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            iterator operator+(size_t) nogil
            iterator operator-(size_t) nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
            bint operator<(iterator) nogil
            bint operator>(iterator) nogil
            bint operator<=(iterator) nogil
            bint operator>=(iterator) nogil
        cppclass reverse_iterator:
            float& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            iterator operator+(size_t) nogil
            iterator operator-(size_t) nogil
            bint operator==(reverse_iterator) nogil
            bint operator!=(reverse_iterator) nogil
            bint operator<(reverse_iterator) nogil
            bint operator>(reverse_iterator) nogil
            bint operator<=(reverse_iterator) nogil
            bint operator>=(reverse_iterator) nogil
        VectorFloat() nogil except +
        VectorFloat(VectorFloat&) nogil except +
        VectorFloat(size_t) nogil except +
        VectorFloat(size_t, float&) nogil except +
        float& operator[](size_t) nogil
        bint operator==(VectorFloat&, VectorFloat&) nogil
        bint operator!=(VectorFloat&, VectorFloat&) nogil
        bint operator<(VectorFloat&, VectorFloat&) nogil
        bint operator>(VectorFloat&, VectorFloat&) nogil
        bint operator<=(VectorFloat&, VectorFloat&) nogil
        bint operator>=(VectorFloat&, VectorFloat&) nogil
        void assign(size_t, float&) nogil
        float& at(size_t) nogil
        float& back() nogil
        iterator begin() nogil
        size_t capacity() nogil
        void clear() nogil
        bint empty() nogil
        iterator end() nogil
        iterator erase(iterator) nogil
        iterator erase(iterator, iterator) nogil
        float& front() nogil
        iterator insert(iterator, float&) nogil
        void insert(iterator, size_t, float&) nogil
        void insert(iterator, iterator, iterator) nogil
        size_t max_size() nogil
        void pop_back() nogil
        void push_back(float&) nogil
        reverse_iterator rbegin() nogil
        reverse_iterator rend() nogil
        void reserve(size_t) nogil
        void resize(size_t) nogil
        void resize(size_t, float&) nogil
        size_t size() nogil
        void swap(VectorFloat&) nogil
        float * data() nogil

    cdef cppclass AlignedFloats:
        AlignedFloats (float *, size_t) nogil
        float * data () nogil
        size_t size () nogil


cdef void vector_float_from_ndarray(
        VectorFloat & vector_float,
        numpy.ndarray[numpy.float32_t, ndim=1] ndarray)


cdef void vector_float_to_ndarray(
        VectorFloat & vector_float,
        numpy.ndarray[numpy.float32_t, ndim=1] ndarray)
