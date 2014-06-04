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
#include <distributions/sparse_counter.hpp>
#include <distributions/vector.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/mixture.hpp>

namespace distributions {
namespace dirichlet_process_discrete {

typedef uint32_t count_t;
typedef uint32_t Value;
struct Group;
struct Scorer;
struct Sampler;
struct VectorizedScorer;
typedef GroupScorerMixture<VectorizedScorer> Mixture;


struct Shared
{
    typedef dirichlet_process_discrete::Value Value;
    typedef dirichlet_process_discrete::Group Group;

    float gamma;
    float alpha;
    float beta0;
    Packed_<float> betas;  // dense

    static constexpr Value OTHER () { return -1; }

    void add_slot (rng_t & rng)
    {
        float beta = beta0 * sample_beta(rng, 1.f, alpha);
        beta0 -= beta;
        betas.packed_add(beta);
    }

    void remove_slot (const Value & value)
    {
        DIST_ASSERT1(value < betas.size(), "value out of bounds: " << value);
        beta0 += betas[value];
        betas.packed_remove(value);
        if (betas.empty()) {
            beta0 = 0;
        }
    }

    static Shared EXAMPLE () {
        Shared shared;
        size_t dim = 100;
        shared.gamma = 0.5;
        shared.alpha = 0.5;
        shared.beta0 = 0.0;  // must be zero for testing
        shared.betas.resize(dim, 1.0 / dim);
        return shared;
    }
};


struct Group
{
    typedef dirichlet_process_discrete::Value Value;

    SparseCounter<Value, count_t> counts;  // sparse

    void init (
            const Shared &,
            rng_t &)
    {
        counts.clear();
    }

    void add_slot (const Shared &)
    {
    }

    void remove_slot (const Shared & shared, const Value & value)
    {
        const Value & last_value = shared.betas.size();
        if (value != last_value) {
            counts.rename(last_value, value);
        }
    }

    void add_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(
            value < shared.betas.size(),
            "value out of bounds: " << value);
        counts.add(value);
    }

    void remove_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(
            value < shared.betas.size(),
            "value out of bounds: " << value);
        counts.remove(value);
    }

    void merge (
            const Shared &,
            const Group & source,
            rng_t &)
    {
        counts.merge(source.counts);
    }

    float score_value (
            const Shared & shared,
            const Value & value,
            rng_t & rng) const;

    float score_data (
            const Shared & shared,
            rng_t &) const
    {
        const size_t size = shared.betas.size();
        const size_t total = counts.get_total();

        float score = 0;
        for (auto i : counts) {
            Value value = i.first;
            DIST_ASSERT1(value < size,
                "unknown value: " << value << " >= " << size);
            float prior_i = shared.betas[value] * shared.alpha;
            score += fast_lgamma(prior_i + i.second)
                   - fast_lgamma(prior_i);
        }
        score += fast_lgamma(shared.alpha)
               - fast_lgamma(shared.alpha + total);

        return score;
    }
};

struct Sampler
{
    std::vector<float> probs;  // dense

    void init (
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        probs.clear();
        probs.reserve(shared.betas.size() + 1);
        for (float beta : shared.betas) {
            probs.push_back(beta * shared.alpha);
        }
        for (auto i : group.counts) {
            probs[i.first] += i.second;
        }
        probs.push_back(shared.beta0 * shared.alpha);

        sample_dirichlet(rng, probs.size(), probs.data(), probs.data());
    }

    Value eval (
            const Shared &,
            rng_t & rng) const
    {
        return sample_discrete(rng, probs.size(), probs.data());
    }
};

struct Scorer
{
    std::vector<float> scores;  // dense

    void init (
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        const size_t size = shared.betas.size();
        const size_t total = group.counts.get_total();
        scores.resize(size);

        const float betas_scale = shared.alpha / (shared.alpha + total);
        for (size_t i = 0; i < size; ++i) {
            scores[i] = betas_scale * shared.betas[i];
        }

        const float counts_scale = 1.0f / (shared.alpha + total);
        for (auto i : group.counts) {
            Value value = i.first;
            DIST_ASSERT(value < size,
                "unknown value: " << value << " >= " << size);
            scores[value] += counts_scale * i.second;
        }
    }

    float eval (
            const Shared &,
            const Value & value,
            rng_t &) const
    {
        size_t size = scores.size();
        DIST_ASSERT(value < size,
            "unknown value: " << value << " >= " << size);
        return fast_log(scores[value]);
    }
};

inline float Group::score_value (
        const Shared & shared,
        const Value & value,
        rng_t & rng) const
{
    Scorer scorer;
    scorer.init(shared, * this, rng);
    return scorer.eval(shared, value, rng);
}

class VectorizedScorer
{
    Packed_<VectorFloat> scores_;  // dense
    VectorFloat scores_shift_;
    mutable VectorFloat temp_;

public:

    typedef dirichlet_process_discrete::Value Value;
    typedef dirichlet_process_discrete::Shared Shared;
    typedef dirichlet_process_discrete::Group Group;
    typedef dirichlet_process_discrete::Scorer BaseScorer;

    void resize (const Shared & shared, size_t size)
    {
        const Value dim = shared.betas.size();
        scores_shift_.resize(size);
        scores_.resize(dim);
        for (Value value = 0; value < dim; ++value) {
            scores_[value].resize(size);
        }
    }

    void add_slot (const Shared & shared)
    {
        const Value value = scores_.size();
        const size_t size = scores_shift_.size();
        const float score = fast_log(shared.alpha * shared.betas[value]);
        scores_.packed_add().resize(size, score);
    }

    void remove_slot (const Shared &, const Value & value)
    {
        DIST_ASSERT1(value < scores_.size(), "value out of bounds: " << value);
        scores_.packed_remove(value);
    }

    void add_group (const Shared & shared, rng_t &)
    {
        scores_shift_.packed_add(0);
        const Value dim = shared.betas.size();
        for (Value value = 0; value < dim; ++value) {
            scores_[value].packed_add(0);
        }
    }

    void remove_group (const Shared & shared, size_t groupid)
    {
        scores_shift_.packed_remove(groupid);
        const Value dim = shared.betas.size();
        for (Value value = 0; value < dim; ++value) {
            scores_[value].packed_remove(groupid);
        }
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t & rng)
    {
        const Value dim = shared.betas.size();
        for (Value value = 0; value < dim; ++value) {
            update_group(shared, groupid, group, value, rng);
        }
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(
            value < shared.betas.size(),
            "value out of bounds: " << value);
        count_t count = group.counts.get_count(value);
        count_t count_sum = group.counts.get_total();
        scores_[value][groupid] =
            fast_log(shared.alpha * shared.betas[value] + count);
        scores_shift_[groupid] = fast_log(shared.alpha + count_sum);
    }

    void update_all (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &)
    {
        const Value dim = shared.betas.size();
        const size_t group_count = slave.groups().size();

        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            const Group & group = slave.groups(groupid);
            for (Value value = 0; value < dim; ++value) {
                scores_[value][groupid] =
                    shared.alpha * shared.betas[value]
                    + group.counts.get_count(value);
            }
            scores_shift_[groupid] = shared.alpha + group.counts.get_total();
        }
        vector_log(group_count, scores_shift_.data());
        for (Value value = 0; value < dim; ++value) {
            vector_log(group_count, scores_[value].data());
        }
    }

    void score_value (
            const Shared & shared,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        DIST_ASSERT1(
            value < shared.betas.size(),
            "value out of bounds: " << value);
        vector_add_subtract(
            scores_accum.size(),
            scores_accum.data(),
            scores_[value].data(),
            scores_shift_.data());
    }

    // not thread safe
    float score_data (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &) const
    {
        const size_t size = shared.betas.size();

        temp_.resize(size + 1);
        for (size_t i = 0; i < size; ++i) {
            temp_[i] = fast_lgamma(shared.betas[i] * shared.alpha);
        }
        temp_.back() = fast_lgamma(shared.alpha);

        float score = 0;
        for (const auto & group : slave.groups()) {
            if (group.counts.get_total()) {
                for (auto i : group.counts) {
                    Value value = i.first;
                    DIST_ASSERT1(value < size,
                        "unknown value: " << value << " >= " << size);
                    float prior_i = shared.betas[value] * shared.alpha;
                    score += fast_lgamma(prior_i + i.second)
                           - temp_[value];
                }
                score += temp_.back()
                       - fast_lgamma(shared.alpha + group.counts.get_total());
            }
        }

        return score;
    }

    void score_data_grid (
            const std::vector<Shared> & shareds,
            const MixtureSlave<Shared> & slave,
            AlignedFloats scores_out,
            rng_t & rng) const
    {
        slave.score_data_grid(shareds, scores_out, rng);
    }
};

inline Value sample_value (
        const Shared & shared,
        const Group & group,
        rng_t & rng)
{
    Sampler sampler;
    sampler.init(shared, group, rng);
    return sampler.eval(shared, rng);
}

} // namespace dirichlet_process_discrete
} // namespace distributions
