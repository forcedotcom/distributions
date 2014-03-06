#include <distributions/special.hpp>

#ifdef USE_INTEL_MKL
#include <mkl.h>
#include <mkl_vml.h>
namespace
{

struct InitializeMKL
{
    InitializeMKL()
    {
        mkl_set_num_threads(1);
        vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }
};
InitializeMKL initialize_mkl;

} // anonymous namespace
#endif // USE_INTEL_MKL

namespace distributions
{

float vector_min (
        const size_t size,
        const float * __restrict__ in)
{
    float res = in[0];
    for (size_t i = 0; i < size; ++i) {
        float x = in[i];
        res = x < res ? x : res;
    }
    return res;
}

float vector_max (
        const size_t size,
        const float * __restrict__ in)
{
    float res = in[0];
    for (size_t i = 0; i < size; ++i) {
        float x = in[i];
        res = x > res ? x : res;
    }
    return res;
}

float vector_sum (
        const size_t size,
        const float * __restrict__ in)
{
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += in[i];
    }
    return res;
}

float vector_dot (
        const size_t size,
        const float * __restrict__ in1,
        const float * __restrict__ in2)
{
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += in1[i] * in2[i];
    }
    return res;
}

void vector_shift (
        const size_t size,
        float * __restrict__ io,
        const float shift)
{
    for (size_t i = 0; i < size; ++i) {
        io[i] += shift;
    }
}

void vector_scale (
        const size_t size,
        float * __restrict__ io,
        const float scale)
{
    for (size_t i = 0; i < size; ++i) {
        io[i] *= scale;
    }
}


void vector_exp (
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

void vector_exp (
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


void vector_log (
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

void vector_log (
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

void vector_lgamma (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out)
{
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_lgamma(in[i]);
    }
}

void vector_lgamma (
        const size_t size,
        float * __restrict__ io)
{
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_lgamma(io[i]);
    }
}


// lgamma_nu(x) = lgamma(x/2 + 1/2) - lgamma(x/2)
void vector_lgamma_nu (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out)
{
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_lgamma_nu(in[i]);
    }
}

void vector_lgamma_nu (
        const size_t size,
        float * __restrict__ io)
{
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_lgamma_nu(io[i]);
    }
}

} // namespace distributions

