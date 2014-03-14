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
#include <distributions/random.hpp>
#include <distributions/timers.hpp>

using namespace distributions;

size_t speedtest (size_t size, size_t iters)
{
    rng_t rng;
    std::vector<float> scores(size);
    for (int i = 0; i < size; ++i){
        scores[i] = 10 * sample_unif01(rng);
    }

    std::vector<float> scores_copy = scores;

    int64_t time = -current_time_us();

    size_t bogus = 0;
    for (int i = 0; i < iters; ++i) {
        bogus = sample_from_scores_overwrite(rng, scores_copy);
        scores_copy = scores;
    }

    time += 2 * current_time_us();

    for (int i = 0; i < iters; ++i) {
        scores_copy = scores;
    }

    time -= current_time_us();

    double time_sec = time * 1e-6;
    double choices_per_sec = size * iters / time_sec;
    std::cout << size << '\t' << choices_per_sec << '\n';

    return bogus;
}

int main()
{
    std::cout << "size" << '\t' << "choices_per_sec" << '\n';

    int max_exponent = 15;
    for (int i = 1; i < max_exponent; ++i) {
        int size = 1 << i;
        int iters = 10 << (max_exponent - i);
        speedtest(size, iters);
    }

    return 0;
}

