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
