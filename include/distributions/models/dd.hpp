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
#include <distributions/vector_math.hpp>
#include <distributions/mixture.hpp>

namespace distributions {
namespace dirichlet_discrete {

typedef uint32_t count_t;
typedef int Value;
template<int max_dim> struct Group;
template<int max_dim> struct Scorer;
template<int max_dim> struct Sampler;
template<int max_dim> struct Mixture;


template<int max_dim>
struct Shared
{
    typedef dirichlet_discrete::Value Value;
    typedef typename dirichlet_discrete::Group<max_dim> Group;
    typedef typename dirichlet_discrete::Scorer<max_dim> Scorer;
    typedef typename dirichlet_discrete::Sampler<max_dim> Sampler;

    int dim;  // fixed parameter
    float alphas[max_dim];  // hyperparamter

    static Shared<max_dim> EXAMPLE ()
    {
        Shared<max_dim> shared;
        shared.dim = max_dim;
        for (int i = 0; i < max_dim; ++i) {
            shared.alphas[i] = 0.5;
        }
        return shared;
    }
};


template<int max_dim>
struct Group
{
    typedef dirichlet_discrete::Value Value;

    count_t count_sum;
    count_t counts[max_dim];

    void init (
            const Shared<max_dim> & shared,
            rng_t &)
    {
        count_sum = 0;
        for (Value value = 0; value < shared.dim; ++value) {
            counts[value] = 0;
        }
    }

    void add_value (
            const Shared<max_dim> & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        count_sum += 1;
        counts[value] += 1;
    }

    void remove_value (
            const Shared<max_dim> & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        count_sum -= 1;
        counts[value] -= 1;
    }

    void merge (
            const Shared<max_dim> & shared,
            const Group<max_dim> & source,
            rng_t &)
    {
        for (Value value = 0; value < shared.dim; ++value) {
            counts[value] += source.counts[value];
        }
    }

    float score_value (
            const Shared<max_dim> & shared,
            const Value & value,
            rng_t & rng) const;
};

template<int max_dim>
struct Sampler
{
    float ps[max_dim];

    void init (
            const Shared<max_dim> & shared,
            const Group<max_dim> & group,
            rng_t & rng)
    {
        for (Value value = 0; value < shared.dim; ++value) {
            ps[value] = shared.alphas[value] + group.counts[value];
        }

        sample_dirichlet(rng, shared.dim, ps, ps);
    }

    Value eval (
            const Shared<max_dim> & shared,
            rng_t & rng) const
    {
        return sample_discrete(rng, shared.dim, ps);
    }
};

template<int max_dim>
struct Scorer
{
    float alpha_sum;
    float alphas[max_dim];

    void init (
            const Shared<max_dim> & shared,
            const Group<max_dim> & group,
            rng_t &)
    {
        alpha_sum = 0;
        for (Value value = 0; value < shared.dim; ++value) {
            float alpha = shared.alphas[value] + group.counts[value];
            alphas[value] = alpha;
            alpha_sum += alpha;
        }
    }

    float eval (
            const Shared<max_dim> & shared,
            const Value & value,
            rng_t &) const
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        return fast_log(alphas[value] / alpha_sum);
    }
};

template<int max_dim>
inline float Group<max_dim>::score_value (
        const Shared<max_dim> & shared,
        const Value & value,
        rng_t & rng) const
{
    Scorer<max_dim> scorer;
    scorer.init(shared, * this, rng);
    return scorer.eval(shared, value, rng);
}


template<int max_dim>
struct VectorizedScorer
{
    typedef dirichlet_discrete::Value Value;
    typedef dirichlet_discrete::Shared<max_dim> Shared;
    typedef dirichlet_discrete::Group<max_dim> Group;
    typedef dirichlet_discrete::Scorer<max_dim> BaseScorer;

    void resize(const Shared & shared, size_t size)
    {
        scores_shift_.resize(size);
        scores_.resize(shared.dim);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value].resize(size);
        }
    }

    void add_group (const Shared & shared, rng_t &)
    {
        scores_shift_.packed_add(0);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value].packed_add(0);
        }
    }

    void remove_group (const Shared & shared, size_t groupid)
    {
        scores_shift_.packed_remove(groupid);
        for (Value value = 0; value < shared.dim; ++value) {
            scores_[value].packed_remove(groupid);
        }
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t & rng)
    {
        for (Value value = 0; value < shared.dim; ++value) {
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
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        scores_[value][groupid] =
            fast_log(shared.alphas[value] + group.counts[value]);
        scores_shift_[groupid] = fast_log(alpha_sum_ + group.count_sum);
    }

    void update_all (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &)
    {
        const size_t group_count = slave.groups().size();

        alpha_sum_ = 0;
        for (Value value = 0; value < shared.dim; ++value) {
            alpha_sum_ += shared.alphas[value];
        }
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            const Group & group = slave.groups()[groupid];
            for (Value value = 0; value < shared.dim; ++value) {
                scores_[value][groupid] =
                    shared.alphas[value] + group.counts[value];
            }
            scores_shift_[groupid] = alpha_sum_ + group.count_sum;
        }
        vector_log(group_count, scores_shift_.data());
        for (Value value = 0; value < shared.dim; ++value) {
            vector_log(group_count, scores_[value].data());
        }
    }

    void score_value(
            const Shared & shared,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        vector_add_subtract(
            scores_accum.size(),
            scores_accum.data(),
            scores_[value].data(),
            scores_shift_.data());
    }

    float score_data (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t & rng) const
    {
        return slave.score_data(shared, rng);
    }

private:

    float alpha_sum_;
    std::vector<VectorFloat> scores_;
    VectorFloat scores_shift_;
};

template<int max_dim>
struct Mixture : public GroupScorerMixture<VectorizedScorer<max_dim>>
{};

template<int max_dim>
inline Value sample_value (
        const Shared<max_dim> & shared,
        const Group<max_dim> & group,
        rng_t & rng)
{
    Sampler<max_dim> sampler;
    sampler.init(shared, group, rng);
    return sampler.eval(shared, rng);
}

template<int max_dim>
inline float score_group (
        const Shared<max_dim> & shared,
        const Group<max_dim> & group,
        rng_t &)
{
    float alpha_sum = 0;
    float score = 0;

    for (Value value = 0; value < shared.dim; ++value) {
        float alpha = shared.alphas[value];
        alpha_sum += alpha;
        score += fast_lgamma(alpha + group.counts[value]) - fast_lgamma(alpha);
    }

    score += fast_lgamma(alpha_sum) - fast_lgamma(alpha_sum + group.count_sum);

    return score;
}

} // namespace dirichlet_discrete
} // namespace distributions
