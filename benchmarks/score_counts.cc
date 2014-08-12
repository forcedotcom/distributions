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
#include <cstdio>
#include <distributions/random.hpp>
#include <distributions/clustering.hpp>
#include <distributions/timers.hpp>

using namespace distributions;  // NOLINT(*)

inline int max(const std::vector<int> & counts) {
    const size_t size = counts.size();
    const int * __restrict__ data = counts.data();

    int result = data[0];
    for (size_t i = 0; i < size; ++i) {
        int value = data[i];
        result = result > value ? result : value;
    }
    return result;
}

size_t speedtest(size_t size, size_t iters, float alpha, float d) {
    Clustering<int>::PitmanYor model;
    model.alpha = alpha;
    model.d = d;

    rng_t rng;
    auto assignments = model.sample_assignments(size, rng);
    std::vector<int> counts(1 + max(assignments), 0);
    for (auto groupid : assignments) {
        ++counts[groupid];
    }

    int64_t time = -current_time_us();

    double bogus = 0;
    for (size_t i = 0; i < iters; ++i) {
        bogus += model.score_counts(counts);
    }

    time += current_time_us();

    double time_sec = time * 1e-6;
    double scores_per_sec = iters / time_sec;
    std::cout <<
        size << '\t' <<
        std::right << std::setw(6) << std::fixed << std::setprecision(1) <<
        max(counts) << '\t' <<
        std::right << std::setw(12) << std::fixed << std::setprecision(1) <<
        scores_per_sec << '\n';

    return bogus;
}

int main(int argc, char ** argv) {
    float alpha = (argc > 1) ? atof(argv[1]) : 1.0f;
    float d = (argc > 2) ? atof(argv[2]) : 0.2f;

    std::cout << "size" << '\t' << "max cat" << '\t' << "scores/sec";
    std::cout << " (alpha = " << alpha << ", d = " << d << ")\n";

    size_t min_exponent = 3;
    size_t max_exponent = 7;
    for (size_t i = min_exponent; i <= max_exponent; ++i) {
        size_t size = size_t(round(pow(10, i)));
        size_t iters = 10000000 / size;
        speedtest(size, iters, alpha, d);
    }

    return 0;
}
