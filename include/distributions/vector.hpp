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

#include <memory>
#include <vector>
#include <distributions/aligned_allocator.hpp>

// DEPRECATED, use DIST_ASSUME_ALIGNED directly
#define VectorFloat_data(vf) (DIST_ASSUME_ALIGNED((vf).data()))

namespace distributions {

template<class Value, class Alloc = std::allocator<Value>>
struct Packed_ : std::vector<Value, Alloc> {
    typedef std::vector<Value, Alloc> Base;

    Packed_() {}
    explicit Packed_(size_t size) : Base(size) {}
    Packed_(size_t size, const Value & value) : Base(size, value) {}

    void packed_remove(size_t pos) {
        DIST_ASSERT1(pos < Base::size(), "bad pos: " << pos);
        if (pos != Base::size() - 1) {
            Base::operator[](pos) = std::move(Base::back());
        }
        Base::pop_back();
    }

    void packed_add(const Value & value) {
        Base::push_back(value);
    }

    Value & packed_add() {
        Base::push_back(Value());
        return Base::back();
    }
};

template<class Value>
class Aligned_ {
 public:
    Aligned_(Value * data, size_t size) :
        data_(data),
        size_(size) {
        DIST_ASSERT_ALIGNED(data_);
    }

    Aligned_(Packed_<Value, aligned_allocator<Value>> & source) :
        data_(source.data()),
        size_(source.size()) {
        if (DIST_DEBUG_LEVEL >= 3) {
            DIST_ASSERT_ALIGNED(data_);
        }
    }

    Value * data() { return data_; }
    size_t size() const { return size_; }
    Value & operator[] (size_t i) { return data_[i]; }

 private:
    Value * const data_;
    const size_t size_;
};

typedef Packed_<float, aligned_allocator<float>> VectorFloat;
typedef Aligned_<float> AlignedFloats;

}  // namespace distributions
