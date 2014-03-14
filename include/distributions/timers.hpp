# pragma once

#include <distributions/common.hpp>
#include <sys/time.h>

namespace distributions
{

inline int64_t current_time_us ()
{
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_usec + 1000000L * t.tv_sec;
}

} // namespace distributions
