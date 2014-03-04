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
