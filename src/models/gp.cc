#include <distributions/models/gp.hpp>
#include <distributions/vector_math.hpp>

namespace distributions
{

void GammaPoisson::classifier_score (
        const Classifier & classifier,
        const Value & value,
        float * scores_accum,
        rng_t &) const
{
    const size_t size = classifier.groups.size();
    const float value_noalias = value;
    float * __restrict__ scores_accum_noalias = scores_accum;
    const float * __restrict__ score = classifier.score.data();
    const float * __restrict__ post_alpha = classifier.post_alpha.data();
    const float * __restrict__ score_coeff = classifier.score_coeff.data();
    float * __restrict__ temp = classifier.temp.data();

    const float log_factorial_value = fast_log_factorial(value);
    for (size_t i = 0; i < size; ++i) {
        temp[i] = fast_lgamma(post_alpha[i] + value_noalias);
    }
    for (size_t i = 0; i < size; ++i) {
        scores_accum[i] += score[i]
            + temp[i]
            - log_factorial_value
            + score_coeff[i] * value_noalias;
    }
}

} // namespace distributions
