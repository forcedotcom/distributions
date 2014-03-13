#include <distributions/models/nich.hpp>
#include <distributions/vector_math.hpp>

namespace distributions
{

void NormalInverseChiSq::classifier_score (
        const Classifier & classifier,
        const Value & value,
        float * scores_accum,
        rng_t &) const
{
    const size_t size = classifier.groups.size();
    const float value_noalias = value;
    float * __restrict__ scores_accum_noalias = scores_accum;
    const float * __restrict__ score = classifier.score.data();
    const float * __restrict__ log_coeff = classifier.log_coeff.data();
    const float * __restrict__ precision = classifier.precision.data();
    const float * __restrict__ mean = classifier.mean.data();
    float * __restrict__ temp = classifier.temp.data();

    for (size_t i = 0; i < size; ++i) {
        temp[i] = 1.f + precision[i] * sqr(value_noalias - mean[i]);
    }
    vector_log(size, temp);
    for (size_t i = 0; i < size; ++i) {
        scores_accum_noalias[i] += score[i] + log_coeff[i] * temp[i];
    }
}

} // namespace distributions
