#pragma once

#include <distributions/special.hpp>

#ifdef USE_INTEL_MKL
#include <mkl_vml.h>
#endif // USE_INTEL_MKL

namespace distributions
{

inline float vector_dot (
        const size_t size,
        const float * __restrict__ x,
        const float * __restrict__ y)
{
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

inline float vector_sum (
        const size_t size,
        const float * __restrict__ x)
{
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += x[i];
    }
    return res;
}

inline void vector_shift (
        const size_t size,
        float * __restrict__ x,
        const float shift)
{
    for (size_t i = 0; i < size; ++i) {
        x[i] += shift;
    }
}

inline void vector_scale (
        const size_t size,
        float * __restrict__ x,
        const float scale)
{
    for (size_t i = 0; i < size; ++i) {
        x[i] *= scale;
    }
}


inline void vector_exp (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out)
{
#ifdef USE_INTEL_MKL
    vsExp(size, in, out);
#else // USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        out[i] = expf(in[i]);
    }
#endif // USE_INTEL_MKL
}

inline void vector_exp (
        const size_t size,
        float * __restrict__ io)
{
#ifdef USE_INTEL_MKL
    vsExp(size, io, io);
#else // USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        io[i] = expf(io[i]);
    }
#endif // USE_INTEL_MKL
}


inline void vector_log (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out)
{
#ifdef USE_INTEL_MKL
    vsLn(size, in, out);
#else // USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_log(in[i]);
    }
#endif // USE_INTEL_MKL
}

inline void vector_log (
        const size_t size,
        float * __restrict__ io)
{
#ifdef USE_INTEL_MKL
    vsLn(size, io, io);
#else // USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_log(io[i]);
    }
#endif // USE_INTEL_MKL
}

inline void vector_lgamma (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out)
{
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_lgamma(in[i]);
    }
}

inline void vector_lgamma (
        const size_t size,
        float * __restrict__ io)
{
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_lgamma(io[i]);
    }
}


// lgamma_nu(x) = lgamma(x/2 + 1/2) - lgamma(x/2)
inline void vector_lgamma_nu (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out)
{
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_lgamma_nu(in[i]);
    }
}

inline void vector_lgamma_nu (
        const size_t size,
        float * __restrict__ io)
{
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_lgamma_nu(io[i]);
    }
}

} // namespace distributions

