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

#include <cmath>
#include <vector>
#include <utility>

#define DIST_ASSERT_CLOSE(x, y) \
    DIST_ASSERT(::distributions::are_close((x), (y)), \
        "expected " #x " close to " #y "; actual " << (x) << " vs " << (y))

namespace distributions {

static const float assert_close_tol = 1e-1f;

template<class T>
inline bool are_close(const T & x, const T & y) {
    return x == y;
}

template<>
inline bool are_close(const float & x, const float & y) {
    return fabs(x - y) <= (1 + fabs(x) + fabs(y)) * assert_close_tol;
}

template<>
inline bool are_close(const double & x, const double & y) {
    return fabs(x - y) <= (1 + fabs(x) + fabs(y)) * assert_close_tol;
}

template<class T, class Alloc>
inline bool are_close(
        const std::vector<T, Alloc> & x,
        const std::vector<T, Alloc> & y) {
    if (x.size() != y.size()) {
        return false;
    }
    for (size_t i = 0; i < x.size(); i++) {
        if (!are_close(x[i], y[i])) {
            return false;
        }
    }
    return true;
}

template<class T1, class T2>
inline bool are_close(
        const std::pair<T1, T2> & x,
        const std::pair<T1, T2> & y) {
    return are_close(x.first, y.first) &&
           are_close(x.second, y.second);
}

}  // namespace distributions
