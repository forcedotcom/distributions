#include <iostream>
#include <iomanip>
#include <distributions/vector.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/timers.hpp>

using namespace distributions;

rng_t rng;

template<class Model>
struct Scorers
{
    struct Group {
        typename Model::Group group;
        typename Model::Scorer scorer;
    };

    std::vector<Group> groups;

    Scorers (
            const Model & model,
            const typename Model::Classifier & classifier)
    {
        const size_t group_count = classifier.groups.size();
        groups.resize(group_count);
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            groups[groupid].group = classifier.groups[groupid];
            model.scorer_init(
                    groups[groupid].scorer,
                    groups[groupid].group,
                    rng);
        }
    }

    void score (
            const Model & model,
            const typename Model::Value & value,
            VectorFloat & scores) const
    {
        const size_t group_count = groups.size();
        for (size_t groupid = 0; groupid < group_count; ++groupid) {
            float score = model.scorer_eval(groups[groupid].scorer, value, rng);
            scores[groupid] += score;
        }
    }
};

template<class Model>
void speedtest (
        const Model & model,
        size_t group_count,
        size_t iters)
{
    typename Model::Classifier classifier;
    classifier.groups.resize(group_count);
    std::vector<typename Model::Value> values;
    for (size_t groupid = 0; groupid < group_count; ++groupid) {
        typename Model::Group & group = classifier.groups[groupid];
        model.group_init(group, rng);
        typename Model::Sampler sampler;
        model.sampler_init(sampler, group, rng);
        for (size_t i = 0; i < 4; ++i) {
            typename Model::Value value = model.sampler_eval(sampler, rng);
            model.group_add_value(group, value, rng);
            values.push_back(value);
        }
    }
    model.classifier_init(classifier, rng);
    Scorers<Model> scorers(model, classifier);
    //std::shuffle(values.begin(), values.end(), rng);  // FIXME
    VectorFloat scores(group_count);

    int64_t time = -current_time_us();
    for (size_t i = 0; i < iters / 8; ++i) {
        vector_zero(scores.size(), scores.data());
        for (size_t j = 0; j < 8; ++j) {
            typename Model::Value value = values[(8 * i + j) % values.size()];
            model.classifier_score(classifier, value, scores, rng);
        }
    }
    time += current_time_us();
    double classifier_rate = iters * 1e3 / time;

    time = -current_time_us();
    for (size_t i = 0; i < iters / 8; ++i) {
        vector_zero(scores.size(), scores.data());
        for (size_t j = 0; j < 8; ++j) {
            typename Model::Value value = values[(8 * i + j) % values.size()];
            scorers.score(model, value, scores);
        }
    }
    time += current_time_us();
    double scorers_rate = iters * 1e3  / time;


    std::cout <<
        Model::short_name() << '\t' <<
        group_count << '\t' <<
        scorers_rate << '\t' <<
        classifier_rate << '\n';
}

template<class Model>
void speedtests (const Model & model)
{
    for (int group_count = 1; group_count <= 1000; group_count *= 10) {
        int iters = 100000 / group_count;
        speedtest(model, group_count, iters);
    }
}

int main()
{
    std::cout <<
        "model" << '\t' <<
        "groups" << '\t' <<
        "scorers" << '\t' <<
        "classifier (rows/ms)" << '\n';

    speedtests(DirichletDiscrete<4>::EXAMPLE());
    speedtests(DirichletProcessDiscrete::EXAMPLE());
    speedtests(NormalInverseChiSq::EXAMPLE());
    speedtests(GammaPoisson::EXAMPLE());

    return 0;
}

