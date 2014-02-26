#pragma once

#include <iostream>

#ifdef THROW_ON_ERROR
#define DIST_ERROR(ARG_message) {\
    std::ostringstream PRIVATE_message; \
    PRIVATE_message \
        << "ERROR " << ARG_message << "\n\t"\
        << __FILE__ << " : " << __LINE__ << "\n\t"\
        << __PRETTY_FUNCTION__ << std::endl; \
    throw std::runtime_error(PRIVATE_message.str()); }
#else // THROW_ON_ERROR
#define DIST_ERROR(ARG_message) {\
    std::cerr << "ERROR " << ARG_message << "\n\t"\
              << __FILE__ << " : " << __LINE__ << "\n\t"\
              << __PRETTY_FUNCTION__ << std::endl; \
    abort(); }
#endif // THROW_ON_ERROR

#define DIST_ASSERT(ARG_cond, ARG_message) \
    { if (not (ARG_cond)) DIST_ERROR(ARG_message) }

namespace distributions
{

int foo ();

} // namespace distributions
