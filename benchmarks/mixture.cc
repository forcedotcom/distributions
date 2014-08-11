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

#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <distributions/vector.hpp>
#include <distributions/models/bb.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/models/bnb.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/timers.hpp>

using namespace distributions;  // NOLINT(*)

rng_t rng;

template<class Model>
struct Scorers {
    struct Group {
        typename Model::Group group;
        typename Model::Scorer scorer;
    };

    std::vector<Group> groups;

    Scorers(
            const typename Model::Shared & shared,
            const typename Model::Mixture & mixture) {
        const size_t group_count = mixture.groups().size();
        groups.resize(group_count);
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            groups[groupid].group = mixture.groups()[groupid];
            groups[groupid].scorer.init(
                    shared,
                    groups[groupid].group,
                    rng);
        }
    }

    void score(
            const typename Model::Shared & shared,
            const typename Model::Value & value,
            VectorFloat & scores) const {
        const size_t group_count = groups.size();
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            float score = groups[groupid].scorer.eval(shared, value, rng);
            scores[groupid] += score;
        }
    }
};

template<class Model>
void speedtest(
        const typename Model::Shared & shared,
        size_t group_count,
        size_t iters) {
    typename Model::Mixture mixture;
    mixture.groups().resize(group_count);
    std::vector<typename Model::Value> values;
    std::vector<size_t> assignments;
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        typename Model::Group & group = mixture.groups()[groupid];
        group.init(shared, rng);
    }
    for (size_t i = 0; i < 4 * group_count; ++i) {
        size_t groupid = sample_int(rng, 0, group_count - 1);
        typename Model::Group & group = mixture.groups()[groupid];
        typename Model::Value value = group.sample_value(shared, rng);
        group.add_value(shared, value, rng);
        values.push_back(value);
        assignments.push_back(groupid);
    }
    mixture.init(shared, rng);
    Scorers<Model> scorers(shared, mixture);
    VectorFloat scores(group_count);

    int64_t time = -current_time_us();
    for (size_t i = 0; i < iters / 8; ++i) {
        vector_zero(scores.size(), scores.data());
        for (size_t j = 0; j < 8; ++j) {
            size_t k = (8 * i + j) % values.size();
            typename Model::Value value = values[k];
            size_t groupid = assignments[k];
            mixture.remove_value(shared, groupid, value, rng);
            mixture.score_value(shared, value, scores, rng);
            mixture.add_value(shared, groupid, value, rng);
        }
    }
    time += current_time_us();
    double mixture_rate = iters * 1e0 / time;

    time = -current_time_us();
    for (size_t i = 0; i < iters / 8; ++i) {
        vector_zero(scores.size(), scores.data());
        for (size_t j = 0; j < 8; ++j) {
            size_t k = (8 * i + j) % values.size();
            typename Model::Value value = values[k];
            size_t groupid = assignments[k];
            typename Scorers<Model>::Group & group = scorers.groups[groupid];
            group.group.remove_value(shared, value, rng);
            group.scorer.init(shared, group.group, rng);
            scorers.score(shared, value, scores);
            group.group.add_value(shared, value, rng);
            group.scorer.init(shared, group.group, rng);
        }
    }
    time += current_time_us();
    double scorers_rate = iters * 1e0  / time;


    std::cout <<
        group_count << '\t' <<
        std::right << std::setw(7) << std::fixed << std::setprecision(2) <<
        scorers_rate << '\t' <<
        std::right << std::setw(7) << std::fixed << std::setprecision(2) <<
        mixture_rate << '\n';
}

template<class Model>
void speedtests() {
    std::cout <<
        demangle(typeid(typename Model::Shared).name()) << '\n' <<
        "Groups" << '\t' <<
        "Scorers" << '\t' <<
        "Mixture (cells/us)" << '\n';

    auto const shared = Model::Shared::EXAMPLE();
    for (int group_count = 1; group_count <= 1000; group_count *= 10) {
        int iters = 500000 / group_count;
        speedtest<Model>(shared, group_count, iters);
    }
}

int main() {
    speedtests<BetaBernoulli>();
    speedtests<DirichletDiscrete<4>>();
    speedtests<DirichletProcessDiscrete>();
    speedtests<GammaPoisson>();
    speedtests<BetaNegativeBinomial>();
    speedtests<NormalInverseChiSq>();

    return 0;
}
