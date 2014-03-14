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

#include <iostream>

#ifdef DIST_THROW_ON_ERROR
#define DIST_ERROR(message) {\
    std::ostringstream PRIVATE_message; \
    PRIVATE_message \
        << "ERROR " << message << "\n\t"\
        << __FILE__ << " : " << __LINE__ << "\n\t"\
        << __PRETTY_FUNCTION__ << std::endl; \
    throw std::runtime_error(PRIVATE_message.str()); }
#else // DIST_THROW_ON_ERROR
#define DIST_ERROR(message) {\
    std::cerr << "ERROR " << message << "\n\t"\
              << __FILE__ << " : " << __LINE__ << "\n\t"\
              << __PRETTY_FUNCTION__ << std::endl; \
    abort(); }
#endif // DIST_THROW_ON_ERROR

#ifndef DIST_DEBUG_LEVEL
#  define DIST_DEBUG_LEVEL 0
#endif // DIST_DEBUG_LEVEL

#define DIST_ASSERT(cond, message) { if (not (cond)) DIST_ERROR(message) }

#define DIST_ASSERT_(level, cond, message) \
    { if (DIST_DEBUG_LEVEL >= (level)) DIST_ASSERT(cond, message) }

#define DIST_ASSERT1(cond, message) DIST_ASSERT_(1, cond, message)
#define DIST_ASSERT2(cond, message) DIST_ASSERT_(2, cond, message)
#define DIST_ASSERT3(cond, message) DIST_ASSERT_(3, cond, message)

namespace distributions
{

enum { SYNCHRONIZE_ENTROPY_FOR_UNIT_TESTING = 1 };

int foo ();

} // namespace distributions
