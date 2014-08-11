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

using namespace distributions;  // NOLINT(*)

size_t speedtest(size_t size, size_t iters) {
    rng_t rng;
    std::vector<float> scores(size);
    for (size_t i = 0; i < size; ++i) {
        scores[i] = 10 * sample_unif01(rng);
    }

    std::vector<float> scores_copy = scores;

    int64_t time = -current_time_us();

    size_t bogus = 0;
    for (size_t i = 0; i < iters; ++i) {
        bogus = sample_from_scores_overwrite(rng, scores_copy);
        scores_copy = scores;
    }

    time += 2 * current_time_us();

    for (size_t i = 0; i < iters; ++i) {
        scores_copy = scores;
    }

    time -= current_time_us();

    double time_us = time;
    double choices_per_us = size * iters / time_us;
    std::cout <<
        size << '\t' <<
        std::right << std::setw(8) << std::fixed << std::setprecision(1) <<
        choices_per_us << '\n';

    return bogus;
}

int main() {
    std::cout << "size" << '\t' << "choices/us" << '\n';

    size_t max_exponent = 15;
    for (size_t i = 1; i < max_exponent; ++i) {
        size_t size = 1 << i;
        size_t iters = 10 << (max_exponent - i);
        speedtest(size, iters);
    }

    return 0;
}

