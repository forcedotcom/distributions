#include <iostream>
#include <distributions/random.hpp>
#include <distributions/timers.hpp>

using namespace distributions;

size_t speedtest (size_t size, size_t iters)
{
    rng_t rng;
    std::vector<float> scores(size);
    for (int i = 0; i < size; ++i){
        scores[i] = 10 * sample_unif01(rng);
    }

    std::vector<float> scores_copy = scores;

    int64_t time = -current_time_us();

    size_t bogus = 0;
    for (int i = 0; i < iters; ++i) {
        bogus = sample_from_scores_overwrite(rng, scores_copy);
        scores_copy = scores;
    }

    time += 2 * current_time_us();

    for (int i = 0; i < iters; ++i) {
        scores_copy = scores;
    }

    time -= current_time_us();

    double time_sec = time * 1e-6;
    double choices_per_sec = size * iters / time_sec;
    std::cout << size << '\t' << choices_per_sec << '\n';

    return bogus;
}

int main()
{
    std::cout << "size" << '\t' << "choices_per_sec" << '\n';

    int max_exponent = 15;
    for (int i = 1; i < max_exponent; ++i) {
        int size = 1 << i;
        int iters = 10 << (max_exponent - i);
        speedtest(size, iters);
    }

    return 0;
}

