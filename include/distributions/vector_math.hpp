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

namespace distributions {
void vector_zero(
        const size_t size,
        float * __restrict__ out);

float vector_min(
        const size_t size,
        const float * __restrict__ in);

float vector_max(
        const size_t size,
        const float * __restrict__ in);

float vector_sum(
        const size_t size,
        const float * __restrict__ in);

float vector_dot(
        const size_t size,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

void vector_shift(
        const size_t size,
        float * __restrict__ io,
        const float shift);

void vector_scale(
        const size_t size,
        float * __restrict__ io,
        const float scale);

// io = -io
void vector_negate(
        const size_t size,
        float * __restrict__ io);

// io += in
void vector_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in);

// io = in - io
void vector_negate_and_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in);

// io += in1 + in2
void vector_add_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

// io += in1 - in2
void vector_add_subtract(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2);
void vector_add_subtract(
        const size_t size,
        float * __restrict__ io,
        const float in1,
        const float * __restrict__ in2);

// io += in1 * in2
void vector_multiply_add(
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

void vector_exp(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_exp(
        const size_t size,
        float * __restrict__ io);

void vector_log(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_log(
        const size_t size,
        float * __restrict__ io);

void vector_lgamma(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_lgamma(
        const size_t size,
        float * __restrict__ io);

// lgamma_nu(x) = lgamma(x/2 + 1/2) - lgamma(x/2)
void vector_lgamma_nu(
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_lgamma_nu(
        const size_t size,
        float * __restrict__ io);

}   // namespace distributions

