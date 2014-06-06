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
