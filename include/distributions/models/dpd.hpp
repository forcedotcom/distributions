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

typedef std::unordered_map<Value, float, TrivialHash<Value>> SparseFloat;

struct Shared
{
    typedef dirichlet_process_discrete::Value Value;
    typedef dirichlet_process_discrete::Group Group;

    float gamma;
    float alpha;
    float beta0;
    Sparse_<Value, float> betas;
    SparseCounter<Value, count_t> counts;

    static constexpr Value OTHER ()
    {
        return std::numeric_limits<Value>::max();
    }

    bool add_value (const Value & value, rng_t & rng)
    {
        LOOM_ASSERT1(value != OTHER(), "cannot add OTHER");
        bool add_group = (counts.add(value) == 1);
        if (DIST_UNLIKELY(add_group)) {
            // FIXME is the following correct?
            float beta = beta0 * sample_beta(rng, 1.f, gamma);
            beta0 = std::max(0.f, beta0 - beta);
            betas.add(value) = beta;
        }

        return add_group;
    }

    bool remove_value (const Value & value)
    {
        LOOM_ASSERT1(value != OTHER(), "cannot remove OTHER");
        bool remove_group = (counts.remove(value) == 0);
        if (DIST_UNLIKELY(remove_group)) {
            beta0 += betas.pop(value);
        }

        return remove_group;
    }

    static Shared EXAMPLE ()
    {
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

    SparseCounter<Value, count_t> counts;

    void init (
            const Shared &,
            rng_t &)
    {
        counts.clear();
    }

    void add_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(shared.betas.contains(value), "unknown value: " << value);
        counts.add(value);
    }

    void remove_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(shared.betas.contains(value), "unknown value: " << value);
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
        const size_t total = counts.get_total();
        const float alpha = shared.alpha;

        float score = 0;
        for (auto & i : counts) {
            Value value = i.first;
            float prior_i = alpha * shared.betas[value];
            score += fast_lgamma(prior_i + i.second)
                   - fast_lgamma(prior_i);
        }
        score += fast_lgamma(alpha)
               - fast_lgamma(alpha + total);

        return score;
    }
};

struct Sampler
{
    std::vector<float> probs;
    std::vector<Value> values;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        probs.clear();
        probs.reserve(shared.betas.size() + 1);
        values.clear();
        values.reserve(shared.betas.size() + 1);
        for (auto & pair : shared.betas) {
            Value value = pair.first;
            float beta = pair.second;
            values.push_back(value);
            probs.push_back(group.get_count(value) + beta * shared.alpha);
        }
        probs.push_back(shared.beta0 * shared.alpha);
        values.push_back(shared.OTHER());

        sample_dirichlet(rng, probs.size(), probs.data(), probs.data());
    }

    Value eval (
            const Shared &,
            rng_t & rng) const
    {
        size_t index = sample_discrete(rng, probs.size(), probs.data());
        return values[index];
    }
};

struct Scorer
{
    Sparse_<Value, float> scores;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        scores.clear();

        const size_t total = group.counts.get_total();
        const float betas_scale = shared.alpha / (shared.alpha + total);
        for (auto & i : shared.betas) {
            scores.add(i.first) = i->second * beta_scale;
        }

        const float counts_scale = 1.0f / (shared.alpha + total);
        for (auto & i : group.counts) {
            scores[i.first] += counts_scale * i.second;
        }

        for (auto & score : scores) {
            score = fast_log(score);
        }
    }

    float eval (
            const Shared &,
            const Value & value,
            rng_t &) const
    {
        return scores[value];
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
    Sparse_<Value, VectorFloat> scores_;
    VectorFloat scores_shift_;
    mutable VectorFloat temp_;

public:

    typedef dirichlet_process_discrete::Value Value;
    typedef dirichlet_process_discrete::Shared Shared;
    typedef dirichlet_process_discrete::Group Group;
    typedef dirichlet_process_discrete::Scorer BaseScorer;

    void resize (const Shared & shared, size_t size)
    {
        scores_shift_.resize(size);
        for (const auto & i : shared.betas) {
            Value value = i.first;
            scores_[value].resize(size);
        }
        if (score_.size() != shared.betas.size()) {
            for (auto & i = scores_.begin(); i != scores_.end();) {
                if (DIST_UNLIKELY(not shared.betas.contains(i->first))) {
                    score_.erase(i++);
                } else {
                    ++i;
                }
            }
        }
    }

    void add_shared_value (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            const Value & value)
    {
        VectorFloat & scores = scores_.add(value);
        const size_t group_count = slave.groups().size();
        const float alpha = shared.alpha;
        const float beta = shared.betas[value];
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            auto count = slave.groups(groupid).counts.get_count(value);
            scores[groupid] = alpha * beta + count;
        }
        vector_log(group_count, scores.data());
    }

    void remove_shared_value (
            const Shared &,
            const Value & value)
    {
        scores_.remove(value);
    }

    void add_group (const Shared & shared, rng_t &)
    {
        for (auto & i : scores_) {
            i.second.packed_add(0);
        }
        scores_shift_.packed_add(0);
    }

    void remove_group (const Shared & shared, size_t groupid)
    {
        for (auto & i : scores_) {
            i->second.packed_remove(groupid);
        }
        scores_shift_.packed_remove(groupid);
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t & rng)
    {
        const float alpha = shared.alpha;
        const count_t count = group.counts.get_count(value);
        for (auto & i : scores_) {
            Value value = i->first;
            i->second[groupid] = fast_log(alpha * shared.betas[value] + count);
        }
        count_t count_sum = group.counts.get_total();
        scores_shift_[groupid] = fast_log(shared.alpha + count_sum);

    }

    void update_value (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value & value,
            rng_t &)
    {
        const float alpha = shared.alpha;
        count_t count = group.counts.get_count(value);
        count_t count_sum = group.counts.get_total();
        scores_.get(value)[groupid] =
            fast_log(alpha * shared.betas[value] + count);
        scores_shift_[groupid] = fast_log(alpha + count_sum);
    }

    void update_all (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &)
    {
        const size_t group_count = slave.groups().size();
        const float alpha = shared.alpha;

        for (auto & i : scores_) {
            Value value = i.first;
            VectorFloat & scores = i->second;
            const float beta = shared.betas[value];
            for (size_t groupid = 0; groupid < group_count; ++groupid) {
                auto count = slave.groups(groupid).counts.get_count(value);
                scores[groupid] = alpha * beta + count;
            }
            vector_log(group_count, scores.data());
        }

        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            auto total = slave.groups(groupid).counts.get_total();
            scores_shift_[groupid] = alpha + total;
        }
        vector_log(group_count, scores_shift_.data());
    }

    void score_value (
            const Shared &,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        vector_add_subtract(
            scores_accum.size(),
            scores_accum.data(),
            scores_.get(value).data(),
            scores_shift_.data());
    }

    // not thread safe
    float score_data (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &) const
    {
        const size_t dim = shared.betas.size();

        temp_.resize(dim + 1);
        for (size_t i = 0; i < dim; ++i) {
            temp_[i] = fast_lgamma(shared.betas[i] * shared.alpha);
        }
        temp_.back() = fast_lgamma(shared.alpha);

        float score = 0;
        for (const auto & group : slave.groups()) {
            if (group.counts.get_total()) {
                for (auto & i : group.counts) {
                    Value value = i.first;
                    DIST_ASSERT1(value < dim,
                        "unknown value: " << value << " >= " << dim);
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
