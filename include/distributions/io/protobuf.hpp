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
#include <distributions/clustering.hpp>
#include <distributions/models/bb.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/models/bnb.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/io/schema.pb.h>

namespace distributions
{

namespace protobuf { using namespace ::protobuf::distributions; }

//----------------------------------------------------------------------------
// Clustering

// FIXME gcc-4.6 fails if these are made template<count_t>

inline void clustering_load (
        typename Clustering<int>::PitmanYor & model,
        const protobuf::Clustering::PitmanYor & message)
{
    model.alpha = message.alpha();
    model.d = message.d();
}
inline void clustering_dump (
        const typename Clustering<int>::PitmanYor & model,
        protobuf::Clustering::PitmanYor & message)
{
    message.set_alpha(model.alpha);
    message.set_d(model.d);
}

inline void clustering_load (
        typename Clustering<int>::PitmanYor & model,
        const protobuf::Clustering & message)
{
    clustering_load(model, message.pitman_yor());
}

inline void clustering_dump (
        const typename Clustering<int>::PitmanYor & model,
        protobuf::Clustering & message)
{
    message.Clear();
    clustering_dump(model, * message.mutable_pitman_yor());
}

inline void clustering_load (
        typename Clustering<int>::LowEntropy & model,
        const protobuf::Clustering::LowEntropy & message)
{
    model.dataset_size = message.dataset_size();
}

inline void clustering_dump (
        const typename Clustering<int>::LowEntropy & model,
        protobuf::Clustering::LowEntropy & message)
{
    message.set_dataset_size(model.dataset_size);
}

inline void clustering_load (
        typename Clustering<int>::LowEntropy & model,
        const protobuf::Clustering & message)
{
    clustering_load(model, message.low_entropy());
}

inline void clustering_dump (
        const typename Clustering<int>::LowEntropy & model,
        protobuf::Clustering & message)
{
    clustering_dump(model, * message.mutable_low_entropy());
}

//----------------------------------------------------------------------------
// Shareds

inline void shared_load (
        BetaBernoulli::Shared & shared,
        const protobuf::BetaBernoulli_Shared & message)
{
    shared.alpha = message.alpha();
    shared.beta = message.beta();
}

inline void shared_dump (
        const BetaBernoulli::Shared & shared,
        protobuf::BetaBernoulli_Shared & message)
{
    message.set_alpha(shared.alpha);
    message.set_beta(shared.beta);
}

template<class Shared>
inline void shared_load (
        Shared & shared,
        const protobuf::DirichletDiscrete_Shared & message)
{
    shared.dim = message.alphas_size();
    DIST_ASSERT_LE(shared.dim, Shared::Model::max_dim);
    for (size_t i = 0; i < shared.dim; ++i) {
        shared.alphas[i] = message.alphas(i);
    }
}

template<class Shared>
inline void shared_dump (
        const Shared & shared,
        protobuf::DirichletDiscrete_Shared & message)
{
    message.Clear();
    for (size_t i = 0; i < shared.dim; ++i) {
        message.add_alphas(shared.alphas[i]);
    }
}

inline void shared_load (
        DirichletProcessDiscrete::Shared & shared,
        const protobuf::DirichletProcessDiscrete_Shared & message)
{
    const size_t value_count = message.values_size();
    DIST_ASSERT_EQ(message.betas_size(), value_count);
    DIST_ASSERT_EQ(message.counts_size(), value_count);
    shared.gamma = message.gamma();
    shared.alpha = message.alpha();
    shared.betas.clear();
    shared.counts.clear();
    double beta_sum = 0;
    for (size_t i = 0; i < value_count; ++i) {
        auto value = message.values(i);
        float beta = message.betas(i);
        DIST_ASSERT_LT(0, beta);
        shared.betas.add(value, beta);
        beta_sum += beta;
        shared.counts.add(value, message.counts(i));
    }
    DIST_ASSERT_LE(beta_sum, 1 + 1e-4);
    shared.beta0 = std::max(0.0, 1.0 - beta_sum);
}

inline void shared_dump (
        const DirichletProcessDiscrete::Shared & shared,
        protobuf::DirichletProcessDiscrete_Shared & message)
{
    message.Clear();
    message.set_gamma(shared.gamma);
    message.set_alpha(shared.alpha);
    for (auto & i : shared.betas) {
        auto value = i.first;
        auto beta = i.second;
        auto count = shared.counts.get_count(value);
        message.add_values(value);
        message.add_betas(beta);
        message.add_counts(count);
    }
}

inline void shared_load (
        GammaPoisson::Shared & shared,
        const protobuf::GammaPoisson_Shared & message)
{
    shared.alpha = message.alpha();
    shared.inv_beta = message.inv_beta();
}

inline void shared_dump (
        const GammaPoisson::Shared & shared,
        protobuf::GammaPoisson_Shared & message)
{
    message.set_alpha(shared.alpha);
    message.set_inv_beta(shared.inv_beta);
}

inline void shared_load (
        BetaNegativeBinomial::Shared & shared,
        const protobuf::BetaNegativeBinomial_Shared & message)
{
    shared.alpha = message.alpha();
    shared.beta = message.beta();
    shared.r = message.r();
}

inline void shared_dump (
        const BetaNegativeBinomial::Shared & shared,
        protobuf::BetaNegativeBinomial_Shared & message)
{
    message.set_alpha(shared.alpha);
    message.set_beta(shared.beta);
    message.set_r(shared.r);
}

inline void shared_load (
        NormalInverseChiSq::Shared & shared,
        const protobuf::NormalInverseChiSq_Shared & message)
{
    shared.mu = message.mu();
    shared.kappa = message.kappa();
    shared.sigmasq = message.sigmasq();
    shared.nu = message.nu();
}

inline void shared_dump (
        const NormalInverseChiSq::Shared & shared,
        protobuf::NormalInverseChiSq_Shared & message)
{
    message.set_mu(shared.mu);
    message.set_kappa(shared.kappa);
    message.set_sigmasq(shared.sigmasq);
    message.set_nu(shared.nu);
}

//----------------------------------------------------------------------------
// Groups

inline void group_load (
        const BetaBernoulli::Shared &,
        BetaBernoulli::Group & group,
        const protobuf::BetaBernoulli::Group & message)
{
    group.heads = message.heads();
    group.tails = message.tails();
}

inline void group_dump (
        const BetaBernoulli::Shared &,
        const BetaBernoulli::Group & group,
        protobuf::BetaBernoulli::Group & message)
{
    message.set_heads(group.heads);
    message.set_tails(group.tails);
}

template<class Shared, class Group>
inline void group_load (
        const Shared & shared,
        Group & group,
        const protobuf::DirichletDiscrete::Group & message)
{
    if (DIST_DEBUG_LEVEL >= 1) {
        DIST_ASSERT_EQ(message.counts_size(), shared.dim);
    }
    group.count_sum = 0;
    for (size_t i = 0; i < shared.dim; ++i) {
        group.count_sum += group.counts[i] = message.counts(i);
    }
}

template<class Shared, class Group>
inline void group_dump (
        const Shared & shared,
        const Group & group,
        protobuf::DirichletDiscrete::Group & message)
{
    message.Clear();
    auto & counts = * message.mutable_counts();
    for (size_t i = 0; i < shared.dim; ++i) {
        counts.Add(group.counts[i]);
    }
}

inline void group_load (
        const DirichletProcessDiscrete::Shared &,
        DirichletProcessDiscrete::Group & group,
        const protobuf::DirichletProcessDiscrete::Group & message)
{
    if (DIST_DEBUG_LEVEL >= 1) {
        DIST_ASSERT_EQ(message.keys_size(), message.values_size());
    }
    group.counts.clear();
    for (size_t i = 0, size = message.keys_size(); i < size; ++i) {
        group.counts.add(message.keys(i), message.values(i));
    }
}

inline void group_dump (
        const DirichletProcessDiscrete::Shared &,
        const DirichletProcessDiscrete::Group & group,
        protobuf::DirichletProcessDiscrete::Group & message)
{
    message.Clear();
    auto & keys = * message.mutable_keys();
    auto & values = * message.mutable_values();
    for (const auto & pair : group.counts) {
        keys.Add(pair.first);
        values.Add(pair.second);
    }
}

inline void group_load (
        const GammaPoisson::Shared &,
        GammaPoisson::Group & group,
        const protobuf::GammaPoisson::Group & message)
{
    group.count = message.count();
    group.sum = message.sum();
    group.log_prod = message.log_prod();
}

inline void group_dump (
        const GammaPoisson::Shared &,
        const GammaPoisson::Group & group,
        protobuf::GammaPoisson::Group & message)
{
    message.set_count(group.count);
    message.set_sum(group.sum);
    message.set_log_prod(group.log_prod);
}

inline void group_load (
        const BetaNegativeBinomial::Shared &,
        BetaNegativeBinomial::Group & group,
        const protobuf::BetaNegativeBinomial::Group & message)
{
    group.count = message.count();
    group.sum = message.sum();
}

inline void group_dump (
        const BetaNegativeBinomial::Shared &,
        const BetaNegativeBinomial::Group & group,
        protobuf::BetaNegativeBinomial::Group & message)
{
    message.set_count(group.count);
    message.set_sum(group.sum);
}

inline void group_load (
        const NormalInverseChiSq::Shared &,
        NormalInverseChiSq::Group & group,
        const protobuf::NormalInverseChiSq::Group & message)
{
    group.count = message.count();
    group.mean = message.mean();
    group.count_times_variance = message.count_times_variance();
}

inline void group_dump (
        const NormalInverseChiSq::Shared &,
        const NormalInverseChiSq::Group & group,
        protobuf::NormalInverseChiSq::Group & message)
{
    message.set_count(group.count);
    message.set_mean(group.mean);
    message.set_count_times_variance(group.count_times_variance);
}

//----------------------------------------------------------------------------
// Grid Priors

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::BetaBernoulli_GridPrior & grid,
        Visitor & visitor)
{
    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();

    for (auto beta : grid.beta()) {
        visitor.add().beta = beta;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::DirichletDiscrete_GridPrior & grid,
        Visitor & visitor)
{
    int dim = visitor.shared().dim;

    for (int i = 0; i < dim; ++i) {
        for (auto alpha : grid.alpha()) {
            visitor.add().alphas[i] = alpha;
        }
        visitor.done();
    }
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::DirichletProcessDiscrete_GridPrior & grid,
        Visitor & visitor)
{
    for (auto gamma : grid.gamma()) {
        visitor.add().gamma = gamma;
    }
    visitor.done();

    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::GammaPoisson_GridPrior & grid,
        Visitor & visitor)
{
    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();

    for (auto inv_beta : grid.inv_beta()) {
        visitor.add().inv_beta = inv_beta;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::BetaNegativeBinomial_GridPrior & grid,
        Visitor & visitor)
{
    for (auto alpha : grid.alpha()) {
        visitor.add().alpha = alpha;
    }
    visitor.done();

    for (auto beta : grid.beta()) {
        visitor.add().beta = beta;
    }
    visitor.done();

    for (auto r : grid.r()) {
        visitor.add().r = r;
    }
    visitor.done();
}

template<class Visitor>
inline void for_each_gridpoint (
        const protobuf::NormalInverseChiSq_GridPrior & grid,
        Visitor & visitor)
{
    for (auto mu : grid.mu()) {
        visitor.add().mu = mu;
    }
    visitor.done();

    for (auto kappa : grid.kappa()) {
        visitor.add().kappa = kappa;
    }
    visitor.done();

    for (auto sigmasq : grid.sigmasq()) {
        visitor.add().sigmasq = sigmasq;
    }
    visitor.done();

    for (auto nu : grid.nu()) {
        visitor.add().nu = nu;
    }
    visitor.done();
}

} // namespace distributions
