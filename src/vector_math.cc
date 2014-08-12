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

#include <distributions/special.hpp>

#if defined  USE_YEPPP

#include <yepBuiltin.h>

#elif defined USE_INTEL_MKL

#include <mkl.h>
#include <mkl_vml.h>
namespace {

struct InitializeMKL {
    InitializeMKL() {
        mkl_set_num_threads(1);
        vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }
};
InitializeMKL initialize_mkl;

}   // anonymous namespace

#endif  // defined USE_YEPPP || defined USE_INTEL_MKL


namespace distributions {

void vector_zero(
        const size_t size,
        float * __restrict__ out) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = 0;
    }
}

float vector_min(
        const size_t size,
        const float * __restrict__ in) {
    float res = in[0];
    for (size_t i = 0; i < size; ++i) {
        float x = in[i];
        res = x < res ? x : res;
    }
    return res;
}

float vector_max(
        const size_t size,
        const float * __restrict__ in) {
    float res = in[0];
    for (size_t i = 0; i < size; ++i) {
        float x = in[i];
        res = x > res ? x : res;
    }
    return res;
}

float vector_sum(
        const size_t size,
        const float * __restrict__ in) {
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += in[i];
    }
    return res;
}

float vector_dot(
        const size_t size,
        const float * __restrict__ in1,
        const float * __restrict__ in2) {
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += in1[i] * in2[i];
    }
    return res;
}

void vector_shift(
        const size_t size,
        float * __restrict__ io,
        const float shift) {
    for (size_t i = 0; i < size; ++i) {
        io[i] += shift;
    }
}

void vector_scale(
        const size_t size,
        float * __restrict__ io,
        const float scale) {
    for (size_t i = 0; i < size; ++i) {
        io[i] *= scale;
    }
}

void vector_negate(
        const size_t size,
        float * __restrict__ io) {
    for (size_t i = 0; i < size; ++i) {
        io[i] = -io[i];
    }
}

void vector_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in) {
    for (size_t i = 0; i < size; ++i) {
        io[i] += in[i];
    }
}

void vector_negate_and_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in) {
    for (size_t i = 0; i < size; ++i) {
        io[i] = in[i] - io[i];
    }
}

void vector_add_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2) {
    for (size_t i = 0; i < size; ++i) {
        io[i] += in1[i] + in2[i];
    }
}

void vector_add_subtract(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2) {
    for (size_t i = 0; i < size; ++i) {
        io[i] += in1[i] - in2[i];
    }
}

void vector_add_subtract(
        const size_t size,
        float * __restrict__ io,
        const float in1,
        const float * __restrict__ in2) {
    for (size_t i = 0; i < size; ++i) {
        io[i] += in1 - in2[i];
    }
}

void vector_multiply_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2) {
    for (size_t i = 0; i < size; ++i) {
        io[i] += in1[i] * in2[i];
    }
}

void vector_exp(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out) {
#if defined USE_INTEL_MKL
    vsExp(size, in, out);
#elif defined USE_YEPPP
    for (size_t i = 0; i < size; ++i) {
        out[i] = yepBuiltin_Exp_32f_32f(in[i]);
    }
#else  // defined USE_YEPPP || defined USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_exp(in[i]);
    }
#endif  // defined USE_YEPPP || defined USE_INTEL_MKL
}

void vector_exp(
        const size_t size,
        float * __restrict__ io) {
#if defined USE_INTEL_MKL
    vsExp(size, io, io);
#elif defined USE_YEPPP
    for (size_t i = 0; i < size; ++i) {
        io[i] = yepBuiltin_Exp_32f_32f(io[i]);
    }
#else  // defined USE_YEPPP || defined USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_exp(io[i]);
    }
#endif  // defined USE_YEPPP || defined USE_INTEL_MKL
}


void vector_log(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out) {
#if defined USE_INTEL_MKL
    vsLn(size, in, out);
// #elif defined USE_YEPPP
//    for (size_t i = 0; i < size; ++i) {
//        out[i] = yepBuiltin_Log_32f_32f(in[i]);
//    }
#else  // defined USE_YEPPP || defined USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_log(in[i]);
    }
#endif  // defined USE_YEPPP || defined USE_INTEL_MKL
}

void vector_log(
        const size_t size,
        float * __restrict__ io) {
#if defined USE_INTEL_MKL
    vsLn(size, io, io);
// #elif defined USE_YEPPP
//    for (size_t i = 0; i < size; ++i) {
//        io[i] = yepBuiltin_Log_32f_32f(io[i]);
//    }
#else  // defined USE_YEPPP || defined USE_INTEL_MKL
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_log(io[i]);
    }
#endif  // defined USE_YEPPP || defined USE_INTEL_MKL
}

void vector_lgamma(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_lgamma(in[i]);
    }
}

void vector_lgamma(
        const size_t size,
        float * __restrict__ io) {
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_lgamma(io[i]);
    }
}


// lgamma_nu(x) = lgamma(x/2 + 1/2) - lgamma(x/2)
void vector_lgamma_nu(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = fast_lgamma_nu(in[i]);
    }
}

void vector_lgamma_nu(
        const size_t size,
        float * __restrict__ io) {
    for (size_t i = 0; i < size; ++i) {
        io[i] = fast_lgamma_nu(io[i]);
    }
}

}   // namespace distributions

