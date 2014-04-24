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
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/io/schema.pb.h>

namespace distributions
{

namespace protobuf { using namespace ::protobuf::distributions; }

//----------------------------------------------------------------------------
// Clustering

// FIXME g++ fails if these are made template<count_t>

inline void clustering_load (
        typename Clustering<int>::PitmanYor & model,
        const protobuf::Clustering::PitmanYor & message)
{
    model.alpha = message.alpha();
    model.d = message.d();
}

inline void clustering_load (
        typename Clustering<int>::PitmanYor & model,
        const protobuf::Clustering & message)
{
    clustering_load(model, message.pitman_yor());
}

inline void clustering_load (
        typename Clustering<int>::LowEntropy & model,
        const protobuf::Clustering::LowEntropy & message)
{
    model.dataset_size = message.dataset_size();
}

template<class count_t>
inline void clustering_load (
        typename Clustering<int>::LowEntropy & model,
        const protobuf::Clustering & message)
{
    clustering_load(model, message.low_entropy());
}

//----------------------------------------------------------------------------
// Shareds

template<int max_dim>
inline void shared_load (
        dirichlet_discrete::Shared<max_dim> & shared,
        const protobuf::DirichletDiscrete & message)
{
    shared.dim = message.alphas_size();
    DIST_ASSERT_LE(shared.dim, max_dim);
    for (size_t i = 0; i < shared.dim; ++i) {
        shared.alphas[i] = message.alphas(i);
    }
}

inline void shared_load (
        dirichlet_process_discrete::Shared & shared,
        const protobuf::DirichletProcessDiscrete & message)
{
    shared.gamma = message.gamma();
    shared.alpha = message.alpha();
    shared.betas.resize(message.betas_size());
    double beta_sum = 0;
    for (size_t i = 0; i < shared.betas.size(); ++i) {
        beta_sum += shared.betas[i] = message.betas(i);
    }
    shared.beta0 = 1 - beta_sum;
}

inline void shared_load (
        gamma_poisson::Shared & shared,
        const protobuf::GammaPoisson & message)
{
    shared.alpha = message.alpha();
    shared.inv_beta = message.inv_beta();
}

inline void shared_load (
        normal_inverse_chi_sq::Shared & shared,
        const protobuf::NormalInverseChiSq & message)
{
    shared.mu = message.mu();
    shared.kappa = message.kappa();
    shared.sigmasq = message.sigmasq();
    shared.nu = message.nu();
}

//----------------------------------------------------------------------------
// Groups

template<int max_dim>
inline void group_load (
        const dirichlet_discrete::Shared<max_dim> & shared,
        dirichlet_discrete::Group<max_dim> & group,
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

template<int max_dim>
inline void group_dump (
        const dirichlet_discrete::Shared<max_dim> & shared,
        const dirichlet_discrete::Group<max_dim> & group,
        protobuf::DirichletDiscrete::Group & message)
{
    message.Clear();
    auto & counts = * message.mutable_counts();
    for (size_t i = 0; i < shared.dim; ++i) {
        counts.Add(group.counts[i]);
    }
}

inline void group_load (
        const dirichlet_process_discrete::Shared &,
        dirichlet_process_discrete::Group & group,
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
        const dirichlet_process_discrete::Shared &,
        const dirichlet_process_discrete::Group & group,
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
        const gamma_poisson::Shared &,
        gamma_poisson::Group & group,
        const protobuf::GammaPoisson::Group & message)
{
    group.count = message.count();
    group.sum = message.sum();
    group.log_prod = message.log_prod();
}

inline void group_dump (
        const gamma_poisson::Shared &,
        const gamma_poisson::Group & group,
        protobuf::GammaPoisson::Group & message)
{
    message.set_count(group.count);
    message.set_sum(group.sum);
    message.set_log_prod(group.log_prod);
}

inline void group_load (
        const normal_inverse_chi_sq::Shared &,
        normal_inverse_chi_sq::Group & group,
        const protobuf::NormalInverseChiSq::Group & message)
{
    group.count = message.count();
    group.mean = message.mean();
    group.count_times_variance = message.count_times_variance();
}

inline void group_dump (
        const normal_inverse_chi_sq::Shared &,
        const normal_inverse_chi_sq::Group & group,
        protobuf::NormalInverseChiSq::Group & message)
{
    message.set_count(group.count);
    message.set_mean(group.mean);
    message.set_count_times_variance(group.count_times_variance);
}

} // namespace distributions
