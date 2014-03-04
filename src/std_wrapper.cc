#include <distributions/std_wrapper.hpp>

namespace std_wrapper
{

rng_t global_rng;

namespace detail
{

normal_distribution_t generate_normal;
chisq_distribution_t generate_chisq;
gamma_distribution_t generate_gamma;
poisson_distribution_t generate_poisson;
std::uniform_real_distribution<double> generate_unif01(0.0, 1.0);

} // namespace detail

} // namespace std_wrapper
