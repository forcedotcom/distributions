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

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>

namespace distributions
{

struct BetaBernoulli
{

//----------------------------------------------------------------------------
// Data

float alphas[2];

//----------------------------------------------------------------------------
// Datatypes

typedef BetaBernoulli Model;

typedef int Value;

struct Group
{
    uint32_t counts[2];
};

struct Sampler
{
    float ps[2];
};

struct Scorer
{
    float alphas[2];
};

struct Classifier
{
    std::vector<Group> groups;
    float alpha_sum;
    std::vector<VectorFloat> scores;
    VectorFloat scores_shift;
};

//----------------------------------------------------------------------------
// Mutation

void group_init (
        Group & group,
        rng_t &) const
{
    group.counts[0] = 0;
    group.counts[1] = 0;
}

void group_add_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(value < 2, "value out of bounds");
    group.counts[value] += 1;
}

void group_remove_value (
        Group & group,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(value < 2, "value out of bounds");
    group.counts[value] -= 1;
}

void group_merge (
        Group & destin,
        const Group & source,
        rng_t &) const
{
    destin.counts[0] += source.counts[0];
    destin.counts[1] += source.counts[1];
}

//----------------------------------------------------------------------------
// Sampling

void sampler_init (
        Sampler & sampler,
        const Group & group,
        rng_t & rng) const
{
    sampler.ps[0] = alphas[0] + group.counts[0];
    sampler.ps[1] = alphas[1] + group.counts[1];
    sample_dirichlet(rng, 2, sampler.ps, sampler.ps);
}

Value sampler_eval (
        const Sampler & sampler,
        rng_t & rng) const
{
    return sample_discrete(rng, 2, sampler.ps);
}

Value sample_value (
        const Group & group,
        rng_t & rng) const
{
    Sampler sampler;
    sampler_init(sampler, group, rng);
    return sampler_eval(sampler, rng);
}

//----------------------------------------------------------------------------
// Scoring

void scorer_init (
        Scorer & scorer,
        const Group & group,
        rng_t &) const
{
    scorer.alphas[0] = alphas[0] + group.counts[0];
    scorer.alphas[1] = alphas[1] + group.counts[1];
}

float scorer_eval (
        const Scorer & scorer,
        const Value & value,
        rng_t &) const
{
    DIST_ASSERT1(value < 2, "value out of bounds");
    return fast_log(
        scorer.alphas[value] / (scorer.alphas[0] + scorer.alphas[1]));
}

float score_value (
        const Group & group,
        const Value & value,
        rng_t & rng) const
{
    DIST_ASSERT1(value < 2, "value out of bounds");
    Scorer scorer;
    scorer_init(scorer, group, rng);
    return scorer_eval(scorer, value, rng);
}

float score_group (
        const Group & group,
        rng_t &) const
{
    return
        + fast_lgamma(alphas[0] + alphas[1])
        - fast_lgamma(scorer.alphas[0] + scorer.alphas[1])
        + fast_lgamma(scorer.alphas[0])
        - fast_lgamma(alphas[0])
        + fast_lgamma(scorer.alphas[1])
        - fast_lgamma(alphas[1])
        ;
}

//----------------------------------------------------------------------------
// Classification

void classifier_init (
        Classifier & classifier,
        rng_t & rng) const
{
}

void classifier_add_group (
        Classifier & classifier,
        rng_t & rng) const
{
}

void classifier_remove_group (
        Classifier & classifier,
        size_t groupid) const
{
}

void classifier_add_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t & rng) const
{
}

void classifier_remove_value (
        Classifier & classifier,
        size_t groupid,
        const Value & value,
        rng_t & rng) const
{
}

void classifier_score (
        const Classifier & classifier,
        const Value & value,
        float * scores_accum,
        rng_t &) const;

}; // struct BetaBernoulli

} // namespace distributions
