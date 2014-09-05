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
#include <distributions/io/schema.pb.h>
#include <distributions/assert_close.hpp>
#include <iostream>

// make cpplint happy
#include <utility>
#include <vector>
#include <string>
#include <algorithm>

namespace google {
namespace protobuf {

inline std::ostream & operator<<(
        std::ostream & o,
        const Message & m) {
    o << "{" << m.ShortDebugString() << "}";
    return o;
}

template<class T>
inline std::ostream & operator<<(
        std::ostream & os,
        const RepeatedField<T> & messages) {
    if (auto size = messages.size()) {
        os << '[' << messages.Get(0);
        for (size_t i = 1; i < size; ++i) {
            os << ',' << messages.Get(i);
        }
        return os << ']';
    } else {
        return os << "[]";
    }
}

template<class T>
inline std::ostream & operator<<(
        std::ostream & os,
        const RepeatedPtrField<T> & messages) {
    if (auto size = messages.size()) {
        os << '[' << messages.Get(0);
        for (size_t i = 1; i < size; ++i) {
            os << ',' << messages.Get(i);
        }
        return os << ']';
    } else {
        return os << "[]";
    }
}

inline bool operator==(
        const Message & x,
        const Message & y) {
    std::string x_string;
    std::string y_string;
    x.SerializeToString(&x_string);
    y.SerializeToString(&y_string);
    return x_string == y_string;
}

template<class T>
inline bool operator==(
        const RepeatedField<T> & x,
        const RepeatedField<T> & y) {
    if (DIST_UNLIKELY(x.size() != y.size())) {
        return false;
    }
    for (size_t i = 0, size = x.size(); i < size; ++i) {
        if (DIST_UNLIKELY(x.Get(i) != y.Get(i))) {
            return false;
        }
    }
    return true;
}

template<class T>
inline bool operator==(
        const RepeatedPtrField<T> & x,
        const RepeatedPtrField<T> & y) {
    if (DIST_UNLIKELY(x.size() != y.size())) {
        return false;
    }
    for (size_t i = 0, size = x.size(); i < size; ++i) {
        if (DIST_UNLIKELY(x.Get(i) != y.Get(i))) {
            return false;
        }
    }
    return true;
}

}  // namespace protobuf
}  // namespace google

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

// protobuf specializations here

template<>
inline bool are_close(
        const protobuf::BetaBernoulli::Shared & x,
        const protobuf::BetaBernoulli::Shared & y) {
    return are_close(x.alpha(), y.alpha()) &&
           are_close(x.beta(), y.beta());
}

template<>
inline bool are_close(
        const protobuf::BetaNegativeBinomial::Shared & x,
        const protobuf::BetaNegativeBinomial::Shared & y) {
    return are_close(x.alpha(), y.alpha()) &&
           are_close(x.beta(), y.beta()) &&
           x.r() == y.r();
}

template<class T>
inline bool are_close(
        const google::protobuf::RepeatedField<T> & x,
        const google::protobuf::RepeatedField<T> & y) {
    if (x.size() != y.size()) {
        return false;
    }
    for (int i = 0; i < x.size(); i++) {
        if (!are_close(x.Get(i), y.Get(i))) {
            return false;
        }
    }
    return true;
}

template<>
inline bool are_close(
        const protobuf::DirichletDiscrete::Shared & x,
        const protobuf::DirichletDiscrete::Shared & y) {
    return are_close(x.alphas(), y.alphas());
}

template<>
inline bool are_close(
        const protobuf::DirichletProcessDiscrete::Shared & x,
        const protobuf::DirichletProcessDiscrete::Shared & y) {
    if (!are_close(x.gamma(), y.gamma()) ||
        !are_close(x.alpha(), y.alpha())) {
        return false;
    }

    if (x.values_size() != y.values_size()) {
        return false;
    }

    // could use std::tuple<uint32_t, float, uint64_t>, but who wants to deal
    // with writing a template specialization for are_close() dealing with the
    // variadic template args?
    //
    // NOTE: the above is actually a fun exercise to try at some point.
    // here's a starting point:
    //     https://github.com/stephentu/silo/blob/master/util.h#L635

    std::vector<std::pair<uint32_t, float>> assoc1_x;
    std::vector<std::pair<uint32_t, float>> assoc1_y;
    std::vector<std::pair<uint32_t, uint64_t>> assoc2_x;
    std::vector<std::pair<uint32_t, uint64_t>> assoc2_y;

    // assumes x, y satisfy the message invariants
    const size_t size = x.values_size();
    for (size_t i = 0; i < size; ++i) {
        assoc1_x.emplace_back(x.values(i), x.betas(i));
        assoc1_y.emplace_back(y.values(i), y.betas(i));
        assoc2_x.emplace_back(x.values(i), x.counts(i));
        assoc2_y.emplace_back(y.values(i), y.counts(i));
    }
    std::sort(assoc1_x.begin(), assoc1_x.end());
    std::sort(assoc1_y.begin(), assoc1_y.end());
    std::sort(assoc2_x.begin(), assoc2_x.end());
    std::sort(assoc2_y.begin(), assoc2_y.end());

    return are_close(assoc1_x, assoc1_y) && assoc2_x == assoc2_y;
}

template<>
inline bool are_close(
        const protobuf::DirichletProcessDiscrete::Group & x,
        const protobuf::DirichletProcessDiscrete::Group & y) {
    if (x.keys_size() != y.keys_size()) {
        return false;
    }
    const size_t size = x.keys_size();
    std::vector<std::pair<uint32_t, uint32_t>> sorted_x(size);
    std::vector<std::pair<uint32_t, uint32_t>> sorted_y(size);
    for (size_t i = 0; i < size; ++i) {
        sorted_x[i].first = x.keys(i);
        sorted_x[i].second = x.values(i);
        sorted_y[i].first = y.keys(i);
        sorted_y[i].second = y.values(i);
    }
    std::sort(sorted_x.begin(), sorted_x.end());
    std::sort(sorted_y.begin(), sorted_y.end());
    return sorted_x == sorted_y;
}

template<>
inline bool are_close(
        const protobuf::GammaPoisson::Shared & x,
        const protobuf::GammaPoisson::Shared & y) {
    return are_close(x.alpha(), y.alpha()) &&
           are_close(x.inv_beta(), y.inv_beta());
}

template<>
inline bool are_close(
        const protobuf::GammaPoisson::Group & x,
        const protobuf::GammaPoisson::Group & y) {
    return x.count() == y.count()
        // and are_close(x.log_prod(), y.log_prod())  // log_prod is very noisy
        and x.sum() == y.sum();
}

template<>
inline bool are_close(
        const protobuf::NormalInverseChiSq::Shared & x,
        const protobuf::NormalInverseChiSq::Shared & y) {
    return are_close(x.mu(), y.mu()) &&
           are_close(x.kappa(), y.kappa()) &&
           are_close(x.sigmasq(), y.sigmasq()) &&
           are_close(x.nu(), y.nu());
}

template<>
inline bool are_close(
        const protobuf::NormalInverseChiSq::Group & x,
        const protobuf::NormalInverseChiSq::Group & y) {
    return x.count() == y.count()
        and are_close(x.mean(), y.mean())
        and are_close(x.count_times_variance(), y.count_times_variance());
}

template<>
inline bool are_close(
        const protobuf::NormalInverseWishart::Shared & x,
        const protobuf::NormalInverseWishart::Shared & y) {
    return are_close(x.mu(), y.mu()) &&
           are_close(x.kappa(), y.kappa()) &&
           are_close(x.psi(), y.psi()) &&
           are_close(x.nu(), y.nu());
}

template<>
inline bool are_close(
        const protobuf::NormalInverseWishart::Group & x,
        const protobuf::NormalInverseWishart::Group & y) {
    return x.count() == y.count() &&
           are_close(x.sum_x(), y.sum_x()) &&
           are_close(x.sum_xxt(), y.sum_xxt());
}

template<>
inline bool are_close(
        const google::protobuf::Message & x,
        const google::protobuf::Message & y) {
    // TODO(fritz) use protobuf reflection to recurse through message structure
    // as in the python distributions.test.util.assert_close
    return x == y;
}

}  // namespace distributions
