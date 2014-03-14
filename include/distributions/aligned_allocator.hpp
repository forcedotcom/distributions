// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

namespace distributions
{

// sse instructions require alignment of 16 bytes
// avx instructions require alignment of 32 bytes
template<class T, int alignment = 32>
class aligned_allocator
{
public:

    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    typedef T * pointer;
    typedef const T * const_pointer;

    typedef T & reference;
    typedef const T & const_reference;

    template <class U>
    aligned_allocator(const aligned_allocator<U, alignment> &) throw() {}
    aligned_allocator (const aligned_allocator &) throw() {}
    aligned_allocator () throw() {}
    ~aligned_allocator () throw() {}

    template<class U>
    struct rebind
    {
        typedef aligned_allocator<U, alignment> other;
    };

    pointer address (reference r) const
    {
        return & r;
    }

    const_pointer address (const_reference r) const
    {
        return & r;
    }

    pointer allocate (size_t n, const void * /* hint */ = 0)
    {
        void * result = NULL;
        if (posix_memalign(& result, alignment, n * sizeof(T))) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(result);
    }

    void deallocate (pointer p, size_type /* count */ )
    {
        free(p);
    }

    void construct (pointer p, const T & val)
    {
        new (p) T(val);
    }

    void destroy (pointer p)
    {
        p->~T();
    }

    size_type max_size () const throw()
    {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }
};

template<class T1, class T2>
inline bool operator== (
        const aligned_allocator<T1> &,
        const aligned_allocator<T2> &) throw()
{
    return true;
}

template<class T1, class T2>
inline bool operator!= (
        const aligned_allocator<T1> &,
        const aligned_allocator<T2> &) throw()
{
    return false;
}

} // namespace distributions
