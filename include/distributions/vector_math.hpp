#pragma once

namespace distributions
{

void vector_zero (
        const size_t size,
        float * __restrict__ out);

float vector_min (
        const size_t size,
        const float * __restrict__ in);

float vector_max (
        const size_t size,
        const float * __restrict__ in);

float vector_sum (
        const size_t size,
        const float * __restrict__ in);

float vector_dot (
        const size_t size,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

void vector_shift (
        const size_t size,
        float * __restrict__ io,
        const float shift);

void vector_scale (
        const size_t size,
        float * __restrict__ io,
        const float scale);

// io += in
void vector_add (
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in);

// io += in1 + in2
void vector_add_add (
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

// io += in1 - in2
void vector_add_subtract (
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

// io += in1 * in2
void vector_multiply_add (
        const size_t size,
        float * __restrict__ io,
        const float * __restrict__ in1,
        const float * __restrict__ in2);

void vector_exp (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_exp (
        const size_t size,
        float * __restrict__ io);

void vector_log (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_log (
        const size_t size,
        float * __restrict__ io);

void vector_lgamma (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_lgamma (
        const size_t size,
        float * __restrict__ io);

// lgamma_nu(x) = lgamma(x/2 + 1/2) - lgamma(x/2)
void vector_lgamma_nu (
        const size_t size,
        const float * __restrict__ in,
        float * __restrict__ out);

void vector_lgamma_nu (
        const size_t size,
        float * __restrict__ io);

} // namespace distributions

