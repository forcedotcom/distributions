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
        DIST_ASSERT1(value < shared.dim, "bad value: out of bounds: " << value);
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
class Mixture
{
public:

    typedef dirichlet_discrete::Value Value;
    typedef dirichlet_discrete::Shared<max_dim> Shared;
    typedef dirichlet_discrete::Group<max_dim> Group;
    typedef dirichlet_discrete::Scorer<max_dim> Scorer;

    std::vector<Group> & groups () { return slave_.groups(); }
    Group & groups (size_t i) { return slave_.groups(i); }
    const std::vector<Group> & groups () const { return slave_.groups(); }
    const Group & groups (size_t i) const { return slave_.groups(i); }

    void init (
            const Shared & shared,
            rng_t & rng)
    {
        slave_.init(shared, rng);
        const auto & groups = slave_.groups();
        const size_t group_count = groups.size();
        scores_shift.resize(group_count);
        alpha_sum = 0;
        scores.resize(shared.dim);
        for (Value value = 0; value < shared.dim; ++value) {
            alpha_sum += shared.alphas[value];
            scores[value].resize(group_count);
        }
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            const Group & group = groups[groupid];
            for (Value value = 0; value < shared.dim; ++value) {
                scores[value][groupid] =
                    shared.alphas[value] + group.counts[value];
            }
            scores_shift[groupid] = alpha_sum + group.count_sum;
        }
        vector_log(group_count, scores_shift.data());
        for (Value value = 0; value < shared.dim; ++value) {
            vector_log(group_count, scores[value].data());
        }
    }

    void add_group (
            const Shared & shared,
            rng_t & rng)
    {
        slave_.add_group(shared, rng);
        scores_shift.packed_add(0);
        for (Value value = 0; value < shared.dim; ++value) {
            scores[value].packed_add(0);
        }
    }

    void remove_group (
            const Shared & shared,
            size_t groupid)
    {
        slave_.remove_group(shared, groupid);
        scores_shift.packed_remove(groupid);
        for (Value value = 0; value < shared.dim; ++value) {
            scores[value].packed_remove(groupid);
        }
    }

    void add_value (
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng)
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        slave_.add_value(shared, groupid, value, rng);
        const Group & group = slave_.groups(groupid);
        scores[value][groupid] =
            fast_log(shared.alphas[value] + group.counts[value]);
        scores_shift[groupid] = fast_log(alpha_sum + group.count_sum);
    }

    void remove_value (
            const Shared & shared,
            size_t groupid,
            const Value & value,
            rng_t & rng)
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        slave_.remove_value(shared, groupid, value, rng);
        const Group & group = slave_.groups(groupid);
        scores[value][groupid] =
            fast_log(shared.alphas[value] + group.counts[value]);
        scores_shift[groupid] = fast_log(alpha_sum + group.count_sum);
    }

    void score_value (
            const Shared & shared,
            const Value & value,
            VectorFloat & scores_accum,
            rng_t &) const
    {
        DIST_ASSERT1(value < shared.dim, "value out of bounds: " << value);
        if (DIST_DEBUG_LEVEL >= 2) {
            DIST_ASSERT_EQ(scores_accum.size(), slave_.groups().size());
        }
        const size_t group_count = slave_.groups().size();
        vector_add_subtract(
            group_count,
            scores_accum.data(),
            scores[value].data(),
            scores_shift.data());
    }

    float score_mixture (
            const Shared & shared,
            rng_t & rng) const
    {
        return slave_.score_mixture(shared, rng);
    }

private:

    MixtureSlave<Shared> slave_;
    float alpha_sum;
    std::vector<VectorFloat> scores;
    VectorFloat scores_shift;
};

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
