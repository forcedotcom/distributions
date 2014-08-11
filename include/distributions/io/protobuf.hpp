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

#include <distributions/io/schema.pb.h>

namespace distributions {

template<class Typename> struct Protobuf;

#define DECLARE_MESSAGE(Typename, decl_, special_, params_)         \
decl_ struct Typename;                                              \
template<special_> struct Protobuf<Typename params_>                \
{                                                                   \
    typedef ::protobuf::distributions::Typename t;                  \
};

DECLARE_MESSAGE(BetaBernoulli, , , )
DECLARE_MESSAGE(DirichletDiscrete,
    template<int max_dim>, int max_dim, <max_dim>)
DECLARE_MESSAGE(DirichletProcessDiscrete, , , )
DECLARE_MESSAGE(GammaPoisson, , , )
DECLARE_MESSAGE(BetaNegativeBinomial, , , )
DECLARE_MESSAGE(NormalInverseChiSq, , , )

#undef DECLARE_MESSAGE

namespace protobuf { using namespace ::protobuf::distributions; }  // NOLINT(*)

}  // namespace distributions
