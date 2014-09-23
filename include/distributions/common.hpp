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

#include <string>
#include <iostream>
#include <sstream>

#ifdef __GNUG__
#  include <cxxabi.h>
#  include <memory>
#endif  // __GNUG__

#ifdef __GNUG__
#  define thread_local __thread
#  define DIST_LIKELY(x) __builtin_expect(!!(x), true)
#  define DIST_UNLIKELY(x) __builtin_expect(!!(x), false)
#else  // __GNUG__
#  warning "ignoring DIST_LIKELY(-), DIST_UNLIKELY(-)"
#  define DIST_LIKELY(x) (x)
#  define DIST_UNLIKELY(x) (x)
#endif  // __GNUG__

#ifdef DIST_THROW_ON_ERROR
#  include <stdexcept>
#  define DIST_ERROR(message) {                         \
    std::ostringstream PRIVATE_message;                 \
    PRIVATE_message                                     \
        << "ERROR " << message << "\n\t"                \
        << __FILE__ << " : " << __LINE__ << "\n\t"      \
        << __PRETTY_FUNCTION__ << '\n';                 \
    throw std::runtime_error(PRIVATE_message.str()); }
#else  // DIST_THROW_ON_ERROR
#  define DIST_ERROR(message) {                         \
    std::ostringstream PRIVATE_message;                 \
    PRIVATE_message                                     \
        << "ERROR " << message << "\n\t"                \
        << __FILE__ << " : " << __LINE__ << "\n\t"      \
        << __PRETTY_FUNCTION__ << '\n';                 \
    std::cerr << PRIVATE_message.str() << std::flush;   \
    abort(); }
#endif  // DIST_THROW_ON_ERROR

#ifdef DIST_DISALLOW_SLOW_FALLBACKS
#  define DIST_THIS_SLOW_FALLBACK_SHOULD_BE_OVERRIDDEN \
    DIST_ERROR("slow fallback has not been overridden");
#else  // DIST_DISALLOW_SLOW_FALLBACKS
#  define DIST_THIS_SLOW_FALLBACK_SHOULD_BE_OVERRIDDEN
#endif  // DIST_DISALLOW_SLOW_FALLBACKS

#define DIST_ASSERT(cond, message) \
    { if (DIST_UNLIKELY(not (cond))) DIST_ERROR(message) }

#define DIST_ASSERT_EQ(x, y) \
    DIST_ASSERT((x) == (y), \
        "expected " #x " == " #y "; actual " << (x) << " vs " << (y))
#define DIST_ASSERT_LE(x, y) \
    DIST_ASSERT((x) <= (y), \
        "expected " #x " <= " #y "; actual " << (x) << " vs " << (y))
#define DIST_ASSERT_LT(x, y) \
    DIST_ASSERT((x) < (y), \
        "expected " #x " < " #y "; actual " << (x) << " vs " << (y))
#define DIST_ASSERT_GE(x, y) \
    DIST_ASSERT((x) >= (y), \
        "expected " #x " >= " #y "; actual " << (x) << " vs " << (y))
#define DIST_ASSERT_GT(x, y) \
    DIST_ASSERT((x) > (y), \
        "expected " #x " > " #y "; actual " << (x) << " vs " << (y))
#define DIST_ASSERT_NE(x, y) \
    DIST_ASSERT((x) != (y), \
        "expected " #x " != " #y "; actual " << (x) << " vs " << (y))

#ifndef DIST_DEBUG_LEVEL
#  define DIST_DEBUG_LEVEL 0
#endif  // DIST_DEBUG_LEVEL

#define DIST_ASSERT_(level, cond, message) \
    { if (DIST_DEBUG_LEVEL >= (level)) DIST_ASSERT(cond, message) }

#define DIST_ASSERT1(cond, message) DIST_ASSERT_(1, cond, message)
#define DIST_ASSERT2(cond, message) DIST_ASSERT_(2, cond, message)
#define DIST_ASSERT3(cond, message) DIST_ASSERT_(3, cond, message)

#ifdef __GNUG__
#  define DIST_ALWAYS_INLINE __attribute__((always_inline))
#  define DIST_NEVER_INLINE __attribute__((never_inline))
#else  // __GNUG__
#  warning "ignoring DIST_ALWAYS_INLINE(-), DIST_NEVER_INLINE(-)"
#  define DIST_ALWAYS_INLINE
#  define DIST_NEVER_INLINE
#endif  // __GNUG__

namespace distributions {

// adapted from http://stackoverflow.com/questions/281818
#ifdef __GNUG__
inline std::string demangle(const char * name) {
    int status = 0;
    std::unique_ptr<char, void(*)(void*)> result {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };
    return (status == 0) ? result.get() : name;
}
#else  // __GNUG__
inline std::string demangle(const char * name) {
    return name;
}
#endif  // __GNUG__

enum { SYNCHRONIZE_ENTROPY_FOR_UNIT_TESTING = 1 };

int foo();

}   // namespace distributions
