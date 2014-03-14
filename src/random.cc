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

#include <distributions/random.hpp>

namespace distributions
{

rng_t global_rng;

void sample_dirichlet (
        rng_t & rng,
        size_t dim,
        const float * alphas,
        float * probs)
{
    float total = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        if (alphas[i] > 0) {
            total += probs[i] = sample_gamma(rng, alphas[i]);
        } else {
            probs[i] = 0;
        }
    }
    float scale = 1.f / total;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] *= scale;
    }
}

void sample_dirichlet_safe (
        rng_t & rng,
        size_t dim,
        const float * alphas,
        float * probs,
        float min_value)
{
    DIST_ASSERT(min_value >= 0, "bad bound: " << min_value);
    float total = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float alpha = alphas[i] + min_value;
        DIST_ASSERT(alpha > 0, "bad alphas[" << i << "] = " << alpha);
        total += probs[i] = sample_gamma(rng, alpha) + min_value;
    }
    float scale = 1.f / total;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] *= scale;
    }
}

//----------------------------------------------------------------------------
// Discrete distribution

float scores_to_likelihoods (std::vector<float> & scores)
{
    const size_t size = scores.size();
    float * __restrict__ scores_data = scores.data();
    float max_score = vector_max(size, scores_data);

    float total = 0;
    for (size_t i = 0; i < size; ++i) {
        total += scores_data[i] = expf(scores_data[i] - max_score);
    }

    return total;
}

float score_from_scores_overwrite (
        rng_t & rng,
        size_t sample,
        std::vector<float> & scores)
{
    const size_t size = scores.size();
    float * __restrict__ scores_data = scores.data();
    float max_score = vector_max(size, scores_data);

    float total = 0;
    for (size_t i = 0; i < size; ++i) {
        total += expf(scores_data[i] -= max_score);
    }

    if (SYNCHRONIZE_ENTROPY_FOR_UNIT_TESTING) {
        sample_unif01(rng);  // consume entropy to match sampler
    }

    float score = scores_data[sample] - log(total);
    return score;
}

} // namespace distributions
