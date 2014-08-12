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

#include <iostream>
#include <iomanip>
#include <distributions/random.hpp>
#include <distributions/timers.hpp>
#include <distributions/aligned_allocator.hpp>
#include <distributions/vendor/fmath.hpp>

#ifdef USE_YEPPP
#include <yepBuiltin.h>
#endif  // USE_YEPPP

#ifdef USE_AMD_LIBM
#include <amdlibm.h>
#endif  // USE_AMD_LIBM

#ifdef USE_INTEL_MKL
#include <mkl_vml.h>
#endif  // USE_INTEL_MKL

using namespace distributions;  // NOLINT(*)

typedef std::vector<float, aligned_allocator<float>> Vector;


struct glibc_exp {
    static const char * name() { return "glibc"; }
    static const char * fun() { return "exp"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = expf(data[i]);
        }
    }
};

struct fmath_exp {
    static const char * name() { return "fmath"; }
    static const char * fun() { return "exp"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = fmath::exp(data[i]);
        }
    }
};

#ifdef USE_YEPPP
struct yeppp_exp {
    static const char * name() { return "yeppp"; }
    static const char * fun() { return "exp"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = yepBuiltin_Exp_32f_32f(data[i]);
        }
    }
};
#endif  // USE_YEPPP

#ifdef USE_AMD_LIBM
struct libm_exp {
    static const char * name() { return "libm"; }
    static const char * fun() { return "exp"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        amd_vrsa_expf(size, data, data);
    }
};
#endif  // USE_AMD_LIBM

#ifdef USE_INTEL_MKL
struct mkl_exp {
    static const char * name() { return "mkl"; }
    static const char * fun() { return "exp"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        vsExp(size, data, data);
    }
};
#endif  // USE_INTEL_MKL


struct glibc_log {
    static const char * name() { return "glibc"; }
    static const char * fun() { return "log"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = logf(data[i]);
        }
    }
};

struct fmath_log {
    static const char * name() { return "fmath"; }
    static const char * fun() { return "log"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = fmath::log(data[i]);
        }
    }
};

struct _eric_log {
    static const char * name() { return "eric"; }
    static const char * fun() { return "log"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = distributions::eric_log(data[i]);
        }
    }
};

#ifdef USE_YEPPP
struct yeppp_log {
    static const char * name() { return "yeppp"; }
    static const char * fun() { return "log"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = yepBuiltin_Log_32f_32f(data[i]);
        }
    }
};
#endif  // USE_YEPPP

#ifdef USE_AMD_LIBM
struct libm_log {
    static const char * name() { return "libm"; }
    static const char * fun() { return "log"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        amd_vrsa_logf(size, data, data);
    }
};
#endif  // USE_AMD_LIBM

#ifdef USE_INTEL_MKL
struct mkl_log {
    static const char * name() { return "mkl"; }
    static const char * fun() { return "log"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        vsLn(size, data, data);
    }
};
#endif  // USE_INTEL_MKL


struct glibc_lgamma {
    static const char * name() { return "glibc"; }
    static const char * fun() { return "lgamma"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = lgammaf(data[i]);
        }
    }
};

struct eric_lgamma {
    static const char * name() { return "eric"; }
    static const char * fun() { return "lgamma"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = fast_lgamma(data[i]);
        }
    }
};

#ifdef USE_INTEL_MKL
struct mkl_lgamma {
    static const char * name() { return "mkl"; }
    static const char * fun() { return "lgamma"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        vsLGamma(size, data, data);
    }
};
#endif  // USE_INTEL_MKL


struct glibc_lgamma_nu {
    static const char * name() { return "glibc"; }
    static const char * fun() { return "lgamma_nu"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = lgammaf(data[i] * 0.5f + 0.5f)
                    - lgammaf(data[i] * 0.5f);
        }
    }
};

struct eric_lgamma_nu {
    static const char * name() { return "eric"; }
    static const char * fun() { return "lgamma_nu"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();
        float * __restrict__ data = & values[0];
        for (size_t i = 0; i < size; ++i) {
            data[i] = fast_lgamma_nu(data[i]);
        }
    }
};

#ifdef USE_INTEL_MKL
struct mkl_lgamma_nu {
    static const char * name() { return "mkl"; }
    static const char * fun() { return "lgamma_nu"; }

    static void inplace(Vector & values) {
        const size_t size = values.size();

        static Vector temp_;
        temp_.resize(size);

        float * __restrict__ data = & values[0];
        float * __restrict__ temp = & temp_[0];

        for (size_t i = 0; i < size; ++i) {
            temp[i] = 0.5f + (data[i] *= 0.5f);
        }

        vsLGamma(size, data, data);
        vsLGamma(size, temp, temp);

        for (size_t i = 0; i < size; ++i) {
            data[i] -= temp[i];
        }
    }
};
#endif  // USE_INTEL_MKL


template<class impl>
void speedtest(size_t size, size_t iters) {
    rng_t rng;
    Vector scores(size);
    for (size_t i = 0; i < size; ++i) {
        scores[i] = 100 * sample_unif01(rng);
    }

    Vector scores_copy = scores;

    int64_t time = -current_time_us();

    for (size_t i = 0; i < iters; ++i) {
        impl::inplace(scores_copy);
        scores_copy = scores;
    }

    time += 2 * current_time_us();

    for (size_t i = 0; i < iters; ++i) {
        scores_copy = scores;
    }

    time -= current_time_us();

    double time_sec = time * 1e-6;
    double ops_per_sec = size * iters / time_sec;
    std::cout
        << std::left << std::setw(10) << impl::fun()
        << std::left << std::setw(8) << impl::name()
        << std::right << std::setw(7) << std::fixed << std::setprecision(1)
            << static_cast<float>(ops_per_sec / 1e6)
        << std::endl;
}

int main() {
#ifdef USE_INTEL_MKL
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#endif  // USE_INTEL_MKL

    const size_t size = 1 << 10;
    const size_t iters = 1 << 13;

    std::cout
        << std::left << std::setw(10) << "Function"
        << std::left << std::setw(8) << "Version"
        << std::right << std::setw(8) << "ops/us"
        << std::endl;

    speedtest<glibc_exp>(size, iters);
    speedtest<fmath_exp>(size, iters);
#ifdef USE_YEPPP
    speedtest<yeppp_exp>(size, iters);
#endif  // USE_YEPPP
#ifdef USE_AMD_LIBM
    speedtest<libm_exp>(size, iters);
#endif  // USE_AMD_LIBM
#ifdef USE_INTEL_MKL
    speedtest<mkl_exp>(size, iters);
#endif  // USE_INTEL_MKL

    std::cout << std::endl;

    speedtest<glibc_log>(size, iters);
    speedtest<fmath_log>(size, iters);
#ifdef USE_YEPPP
    speedtest<yeppp_log>(size, iters);
#endif  // USE_YEPPP
#ifdef USE_AMD_LIBM
    speedtest<libm_log>(size, iters);
#endif  // USE_AMD_LIBM
#ifdef USE_INTEL_MKL
    speedtest<mkl_log>(size, iters);
#endif  // USE_INTEL_MKL
    speedtest<_eric_log>(size, iters);

    std::cout << std::endl;

    speedtest<glibc_lgamma>(size, iters);
#ifdef USE_INTEL_MKL
    speedtest<mkl_lgamma>(size, iters);
#endif  // USE_INTEL_MKL
    speedtest<eric_lgamma>(size, iters);

    std::cout << std::endl;

    speedtest<glibc_lgamma_nu>(size, iters);
#ifdef USE_INTEL_MKL
    speedtest<mkl_lgamma_nu>(size, iters);
#endif  // USE_INTEL_MKL
    speedtest<eric_lgamma_nu>(size, iters);

    return 0;
}
