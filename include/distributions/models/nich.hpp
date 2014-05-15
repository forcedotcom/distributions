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
#include <distributions/mixture.hpp>

namespace distributions {
namespace normal_inverse_chi_sq {

typedef float Value;
struct Group;
struct Scorer;
struct Sampler;
struct VectorizedScorer;
typedef GroupScorerMixture<VectorizedScorer> Mixture;


struct Shared
{
    typedef normal_inverse_chi_sq::Value Value;
    typedef normal_inverse_chi_sq::Group Group;

    float mu;
    float kappa;
    float sigmasq;
    float nu;

    Shared plus_group(const Group & group) const;

    static Shared EXAMPLE ()
    {
        Shared shared;
        shared.mu = 0.0;
        shared.kappa = 1.0;
        shared.sigmasq = 1.0;
        shared.nu = 1.0;
        return shared;
    }
};


struct Group
{
    typedef normal_inverse_chi_sq::Value Value;

    uint32_t count;
    float mean;
    float count_times_variance;

    void init (
            const Shared &,
            rng_t &)
    {
        count = 0;
        mean = 0.f;
        count_times_variance = 0.f;
    }

    void add_value (
            const Shared &,
            const Value & value,
            rng_t &)
    {
        ++count;
        float delta = value - mean;
        mean += delta / count;
        count_times_variance += delta * (value - mean);
    }

    void remove_value (
            const Shared &,
            const Value & value,
            rng_t &)
    {
        float total = mean * count;
        float delta = value - mean;
        DIST_ASSERT(count > 0, "Can't remove empty group");

        --count;
        if (count == 0) {
            mean = 0.f;
        } else {
            mean = (total - value) / count;
        }
        if (count <= 1) {
            count_times_variance = 0.f;
        } else {
            count_times_variance -= delta * (value - mean);
        }
    }

    void merge (
            const Shared &,
            const Group & source,
            rng_t &)
    {
        uint32_t total_count = count + source.count;
        float delta = source.mean - mean;
        float source_part = float(source.count) / total_count;
        float cross_part = count * source_part;
        count = total_count;
        mean += source_part * delta;
        count_times_variance +=
            source.count_times_variance + cross_part * sqr(delta);
    }

    float score_value (
            const Shared & shared,
            const Value & value,
            rng_t & rng) const;

    float score_data (
            const Shared & shared,
            rng_t &) const
    {
        Shared post = shared.plus_group(*this);
        float log_pi = 1.1447298858493991f;
        float score = fast_lgamma(0.5f * post.nu)
                    - fast_lgamma(0.5f * shared.nu);
        score += 0.5f * fast_log(shared.kappa / post.kappa);
        score += 0.5f * shared.nu * (fast_log(shared.nu * shared.sigmasq))
               - 0.5f * post.nu * fast_log(post.nu * post.sigmasq);
        score += -0.5f * count * log_pi;
        return score;
    }
};

struct Sampler
{
    float mu;
    float sigmasq;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        Shared post = shared.plus_group(group);
        sigmasq = post.nu * post.sigmasq / sample_chisq(rng, post.nu);
        mu = sample_normal(rng, post.mu, sigmasq / post.kappa);
    }

    Value eval (
            const Shared &,
            rng_t & rng) const
    {
        return sample_normal(rng, mu, sigmasq);
    }
};

struct Scorer
{
    float score;
    float log_coeff;
    float precision;
    float mean;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        Shared post = shared.plus_group(group);
        float lambda = post.kappa / ((post.kappa + 1.f) * post.sigmasq);
        score = fast_lgamma_nu(post.nu)
              + 0.5f * fast_log(lambda / (M_PIf * post.nu));
        log_coeff = -0.5f * post.nu - 0.5f;
        precision = lambda / post.nu;
        mean = post.mu;
    }

    float eval (
            const Shared &,
            const Value & value,
            rng_t &) const
    {
        return score
             + log_coeff * fast_log(
                 1.f + precision * sqr(value - mean));
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

struct VectorizedScorer
{
    typedef normal_inverse_chi_sq::Value Value;
    typedef normal_inverse_chi_sq::Shared Shared;
    typedef normal_inverse_chi_sq::Group Group;
    typedef normal_inverse_chi_sq::Scorer BaseScorer;

    void resize(const Shared &, size_t size)
    {
        score_.resize(size);
        log_coeff_.resize(size);
        precision_.resize(size);
        mean_.resize(size);
        temp_.resize(size);
    }

    void add_group (const Shared &, rng_t &)
    {
        score_.packed_add();
        log_coeff_.packed_add();
        precision_.packed_add();
        mean_.packed_add();
        temp_.packed_add();
    }

    void remove_group (const Shared &, size_t groupid)
    {
        score_.packed_remove(groupid);
        log_coeff_.packed_remove(groupid);
        precision_.packed_remove(groupid);
        mean_.packed_remove(groupid);
        temp_.packed_remove(groupid);
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            rng_t & rng)
    {
        BaseScorer base;
        base.init(shared, group, rng);

        score_[groupid] = base.score;
        log_coeff_[groupid] = base.log_coeff;
        precision_[groupid] = base.precision;
        mean_[groupid] = base.mean;
    }

    void update_group (
            const Shared & shared,
            size_t groupid,
            const Group & group,
            const Value &,
            rng_t & rng)
    {
        update_group(shared, groupid, group, rng);
    }

    void update_all (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t & rng)
    {
        const size_t group_count = slave.groups().size();
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            update_group(shared, groupid, slave.groups()[groupid], rng);
        }
    }

    // not thread safe
    void score_value (
            const Shared &,
            const Value & value,
            AlignedFloats scores_accum,
            rng_t &) const;

    float score_data (
            const Shared & shared,
            const MixtureSlave<Shared> & slave,
            rng_t &) const
    {
        const float nu_part = fast_lgamma(0.5f * shared.nu);
        const float sigmasq_part =
            0.5f * shared.nu * fast_log(shared.nu * shared.sigmasq);
        const float log_pi = 1.1447298858493991f;

        float score = 0;
        for (const auto & group : slave.groups()) {
            if (group.count) {
                Shared post = shared.plus_group(group);
                score += fast_lgamma(0.5f * post.nu) - nu_part;
                score += 0.5f * fast_log(shared.kappa / post.kappa);
                score += sigmasq_part
                       - 0.5f * post.nu * fast_log(post.nu * post.sigmasq);
                score += -0.5f * log_pi * group.count;
            }
        }

        return score;
    }

private:

    VectorFloat score_;
    VectorFloat log_coeff_;
    VectorFloat precision_;
    VectorFloat mean_;
    mutable VectorFloat temp_;
};

inline Shared Shared::plus_group (const Group & group) const
{
    Shared post;
    float mu_1 = mu - group.mean;
    post.kappa = kappa + group.count;
    post.mu = (kappa * mu + group.mean * group.count) / post.kappa;
    post.nu = nu + group.count;
    post.sigmasq = 1.f / post.nu * (
        nu * sigmasq
        + group.count_times_variance
        + (group.count * kappa * mu_1 * mu_1) / post.kappa);
    return post;
}

inline Value sample_value (
        const Shared & shared,
        const Group & group,
        rng_t & rng)
{
    Sampler sampler;
    sampler.init(shared, group, rng);
    return sampler.eval(shared, rng);
}

} // namespace normal_inverse_chi_sq
} // namespace distributions
