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

from libc.stdint cimport uint32_t


cdef extern from "distributions/mixture.hpp":
    cppclass MixtureIdTracker_cc "distributions::MixtureIdTracker":
        void init (size_t group_count) nogil except +
        void add_group () nogil except +
        void remove_group (uint32_t packed) nogil except +
        uint32_t packed_to_global (uint32_t packed) nogil except +
        uint32_t global_to_packed (uint32_t packed) nogil except +


cdef class MixtureIdTracker:
    cdef MixtureIdTracker_cc * ptr
    def __cinit__(self):
        self.ptr = new MixtureIdTracker_cc()
    def __dealloc__(self):
        del self.ptr

    def init(self, int group_count=0):
        self.ptr.init(group_count)

    def add_group(self):
        self.ptr.add_group()

    def remove_group(self, int packed):
        self.ptr.remove_group(packed)

    def packed_to_global(self, int packed):
        return self.ptr.packed_to_global(packed)

    def global_to_packed(self, int global_):
        return self.ptr.global_to_packed(global_)
