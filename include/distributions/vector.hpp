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

#include <distributions/aligned_allocator.hpp>
#include <vector>

// DEPRECATED, use DIST_ASSUME_ALIGNED directly
#define VectorFloat_data(vf) (DIST_ASSUME_ALIGNED(vf.data()))

namespace distributions
{

typedef std::vector<float, aligned_allocator<float>> VectorFloatBase;

class ArrayFloat
{
public:

    ArrayFloat (float * data, size_t size) :
        data_(data),
        size_(size)
    {
        DIST_ASSERT_ALIGNED(data_);
    }

    ArrayFloat (VectorFloatBase & source) :
        data_(source.data()),
        size_(source.size())
    {
        if (DIST_DEBUG_LEVEL >= 3) {
            DIST_ASSERT_ALIGNED(data_);
        }
    }

    float * data () { return data_; }
    size_t size () const { return size_; }

private:

    float * const data_;
    const size_t size_;
};

struct VectorFloat : VectorFloatBase
{
    typedef VectorFloatBase Base;

    VectorFloat () {}
    VectorFloat (size_t size) : Base(size) {}
    VectorFloat (size_t size, float value) : Base(size, value) {}

    operator ArrayFloat () { return ArrayFloat(* this); }

    void packed_remove (size_t pos)
    {
        DIST_ASSERT1(pos < Base::size(), "bad pos: " << pos);
        if (DIST_LIKELY(pos != Base::size() - 1)) {
            Base::operator[](pos) = Base::back();
        }
        Base::pop_back();
    }

    void packed_add (const float & value)
    {
        Base::push_back(value);
    }

    float & packed_add ()
    {
        Base::push_back(0);
        return Base::back();
    }

    void padded_resize (size_t size, float fill = 0)
    {
        Base::resize((size + 7) / 8 * 8, fill);
    }
};

} // namespace distributions
