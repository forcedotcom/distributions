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
#include <cstdio>
#include <distributions/random.hpp>
#include <distributions/clustering.hpp>
#include <distributions/timers.hpp>

using namespace distributions;

inline int max (const std::vector<int> & counts)
{
    const size_t size = counts.size();
    const int * __restrict__ data = counts.data();

    int result = data[0];
    for (size_t i = 0; i < size; ++i) {
        int value = data[i];
        result = result > value ? result : value;
    }
    return result;
}

size_t speedtest (size_t size, size_t iters, float alpha, float d)
{
    Clustering<int>::PitmanYor model;
    model.alpha = alpha;
    model.d = d;

    rng_t rng;

    int64_t time = -current_time_us();

    size_t bogus = 0;
    double total_cats = 0;
    for (int i = 0; i < iters; ++i) {
        total_cats += max(model.sample_assignments(size, rng));
    }

    time += current_time_us();

    double time_sec = time * 1e-6;
    double samples_per_sec = iters / time_sec;
    double mean_cats = total_cats / iters;
    std::cout <<
        size << '\t' <<
        mean_cats << '\t' <<
        samples_per_sec << '\n';

    return bogus;
}

int main (int argc, char ** argv)
{
    float alpha = (argc > 1) ? atof(argv[1]) : 1.0f;
    float d = (argc > 2) ? atof(argv[2]) : 0.2f;

    std::cout << "size" << '\t' << "cats" << '\t' << "samples_per_sec";
    std::cout << " (alpha = " << alpha << ", d = " << d << ")\n";

    int min_exponent = 10;
    int max_exponent = 20;
    for (int i = min_exponent; i <= max_exponent; ++i) {
        int size = 1 << i;
        int iters = 4 << (max_exponent - i);
        speedtest(size, iters, alpha, d);
    }

    return 0;
}
