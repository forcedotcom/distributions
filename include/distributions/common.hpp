#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <cstring>
#include <iostream>

#define M_PIf (3.14159265358979f)

#if 1
#define DISTRIBUTIONS_ERROR(ARG_message) {\
    std::cerr << "ERROR " << ARG_message << "\n\t"\
              << __FILE__ << " : " << __LINE__ << "\n\t"\
              << __PRETTY_FUNCTION__ << std::endl; \
    abort(); }
#else
#define DISTRIBUTIONS_ERROR(ARG_message) {\
    std::ostringstream PRIVATE_message; \
    PRIVATE_message \
        << "ERROR " << ARG_message << "\n\t"\
        << __FILE__ << " : " << __LINE__ << "\n\t"\
        << __PRETTY_FUNCTION__ << std::endl; \
    throw std::runtime_error(PRIVATE_message.str()); }
#endif

#define DISTRIBUTIONS_ASSERT(ARG_cond, ARG_message) \
    { if (not (ARG_cond)) DISTRIBUTIONS_ERROR(ARG_message) }

namespace distributions
{

int foo ();


typedef std::mt19937 rng_t;


/// Implements the ICSI fast log algorithm, v2.
class FastLog
{
public:

    FastLog (int N);

    inline float log (float x)
    {
        //int intx = * reinterpret_cast<int *>(& x);
        int intx;
        memcpy(&intx, &x, 4);

        register const int exp = ((intx >> 23) & 255) - 127;
        register const int man = (intx & 0x7FFFFF) >> (23 - N_);

        // exponent plus lookup refinement
        return ((float)(exp) + table_[man]) * 0.69314718055994529f;
    }

private:

    const int N_;
    std::vector<float> table_;
};

static FastLog GLOBAL_FAST_LOG_14(14);

inline float fastlog (float x)
{
    return GLOBAL_FAST_LOG_14.log(x);
}


extern const char LogTable256[256];
extern const float lgamma_approx_coeff5[]; 

inline float lgamma_approx_positive5 (float y)
{
    // A piecewise fifth-order approximation of loggamma,
    // which bottoms out in libc gammaln for vals < 1.0
    // and throws an exception outside of the domain 2**32
    //
    // see loggamma.py for the code used to generate the coefficient table

    DISTRIBUTIONS_ASSERT(y <= 4294967295.0,
        "loggamma approx : value " << y << " outside of domain");

    if (y < 2.5) {
        return lgamma(y);
    }

    // stolen from:
    // http://www-graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
    float v = y;                // find int(log2(v)), where v > 0.0 && finite(v)
    int c;                      // 32-bit int c gets the result;
    int x = *(const int *) &v;  // or portably:  memcpy(&x, &v, sizeof x);

    c = x >> 23;

    if (c) {
        c -= 127;
    } else { // subnormal, so recompute using mantissa: c = intlog2(x) - 149;
        register unsigned int t; // temporary
        if ((t = x >> 16)) {
            c = LogTable256[t] - 133;
        } else {
            c = (t = x >> 8) ? LogTable256[t] - 141 : LogTable256[x] - 149;
        }
    }

    int pos = c *6;
    float a5 = lgamma_approx_coeff5[pos];
    float a4 = lgamma_approx_coeff5[pos + 1];
    float a3 = lgamma_approx_coeff5[pos + 2];
    float a2 = lgamma_approx_coeff5[pos + 3];
    float a1 = lgamma_approx_coeff5[pos + 4];
    float a0 = lgamma_approx_coeff5[pos + 5];

    double yprod = y;
    double sum = a0;
    sum += a1 * yprod;

    yprod *= y;
    sum += a2 * yprod;

    yprod *= y;
    sum += a3 * yprod;

    yprod *= y;
    sum += a4 * yprod;

    yprod *= y;
    sum += a5 * yprod;

    return sum;
}

} // namespace distributions
