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

#include <distributions/special.hpp>
#include <distributions/vector.hpp>
#include <mutex>

namespace distributions {
namespace detail {

FastLog::FastLog(int N)
    : N_(N),
      table_(1 << N) {
    for (int i = 0; i < (1 << N_); i++) {
        float v = 1.0
                + static_cast<float>(
                    static_cast<float>(i) * (1 << (23 - N_))) / (1 << 23);
        table_[i] = log2(v);
    }
}

const char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
#undef LT
};


static std::mutex log_stirling1_mutex;
static std::vector<VectorFloat> log_stirling1_cache;

inline void log_stirling1_cache_add() {
    size_t n = log_stirling1_cache.size();
    log_stirling1_cache.push_back(VectorFloat());
    VectorFloat & row = log_stirling1_cache.back();
    row.resize(n + 1);
    row[0] = -INFINITY;
    row[n] = 0;
    if (n > 1) {
        const VectorFloat & prev = log_stirling1_cache[n - 1];
        const float log_n_minus_1 = logf(n - 1);
        for (size_t k = 1; k < n; ++k) {
            row[k] = log_sum_exp(log_n_minus_1 + prev[k], prev[k - 1]);
        }
    }
}

inline void get_log_stirling1_row_exact(const size_t n, float * row) {
    // this could really be a readers-writer shared_mutex
    std::unique_lock<std::mutex> lock(log_stirling1_mutex);

    while (n >= log_stirling1_cache.size()) {
        log_stirling1_cache_add();
    }

    auto const & cached = log_stirling1_cache[n];
    memcpy(row, cached.data(), cached.size() * sizeof(row[0]));
}

inline void get_log_stirling1_row_approx(const size_t n, float * row) {
    // Approximation #1 is taken from Eqn 26.8.40 of [1],
    // whose unsigned version is
    //
    //                 n!
    //   s(n+1, k+1) = -- (gamma + log(n))^k            (approx1)
    //                 k!
    //
    // where gamma is the Euler-Mascheroni constant.
    // Approximation #2 is taken from pp. 6 of [2]
    //
    //            (n^2 / 2)^(n-r)
    //   s(n,r) = ---------------                       (approx2)
    //               n! (n-r)!
    //
    // where Gruenberg's r is our k and Gruenberg's S_{r,1,n} is our s(n,k).
    // Approximation #1 is accurate for small k, and approximation #2 is
    // accurate for large k; both are overestimates.
    // We use a softmin of these two approximations,
    // with an ad hoc softness that depends on n.
    //
    // [1] "Digital Library of Mathematical Functions"
    //   http://dlmf.nist.gov/26.8
    // [2] "On asymptotics, Stirling numbers, Gamma function and polylogs"
    //   -Daniel B. Gruenberg
    //   http://arxiv.org/abs/math/0607514

    const float log_factorial_n_minus_1 = fast_log_factorial(n - 1);
    const float log_n_squared_over_two = logf(n * n / 2.0f);
    const float euler_gamma = 0.57721566490153286060f;
    const float log_stuff = logf(euler_gamma + logf(n - 1));
    const float softness = n / 3.0;  // ad hoc

    // endpoints
    row[0] = -INFINITY;
    row[n] = 0;

    // internal points
    for (size_t k = 1; k < n; ++k) {
        float approx1 = log_factorial_n_minus_1
                      - fast_log_factorial(k - 1)
                      + (k - 1) * log_stuff;
        float approx2 = (n - k) * log_n_squared_over_two
                      - fast_log_factorial(n - k);
        row[k] = -softness * fast_log_sum_exp(
            -1.0f / softness * approx1,
            -1.0f / softness * approx2);
    }
}

void get_log_stirling1_row(size_t n, float * result) {
    if (n < 32) {
        get_log_stirling1_row_exact(n, result);
    } else {
        get_log_stirling1_row_approx(n, result);
    }
}

const float lgamma_approx_coeff5[] = {
-3.29075828194618e-02, 3.11402469873428e-01, -1.26565241813660e+00,
3.06901979446411e+00, -3.99838900566101e+00, 1.91650712490082e+00,
-1.27542507834733e-03, 2.45208702981472e-02, -2.05080837011337e-01,
1.06328070163727e+00, -2.05143237113953e+00, 1.13884496688843e+00,
-5.90835916227661e-05, 2.30360287241638e-03, -3.94700765609741e-02,
4.31111842393875e-01, -8.11912357807159e-01, 1.38451874256134e-01,
-3.13346959046612e-06, 2.46525305556133e-04, -8.57665389776230e-03,
1.93337082862854e-01, 1.28577262163162e-01, -1.39292562007904e+00,
-1.79876764150322e-07, 2.84436719084624e-05, -1.99566851370037e-03,
9.15164873003960e-02, 9.38368678092957e-01, -4.04415321350098e+00,
-1.07678470584460e-08, 3.41422332894581e-06, -4.81184630189091e-04,
4.45219948887825e-02, 1.68803942203522e+00, -8.96704387664795e+00,
-6.58549492715821e-10, 4.18171737237572e-07, -1.18131858471315e-04,
2.19584535807371e-02, 2.40899610519409e+00, -1.84496879577637e+01,
-4.07141056979476e-11, 5.17405069899723e-08, -2.92657005047658e-05,
1.09044071286917e-02, 3.11593461036682e+00, -3.70601158142090e+01,
-2.53081214839079e-12, 6.43459463489648e-09, -7.28322174836649e-06,
5.43360086157918e-03, 3.81594920158386e+00, -7.39302520751953e+01,
-1.57745385222056e-13, 8.02270583299958e-10, -1.81666780463274e-06,
2.71216244436800e-03, 4.51252317428589e+00, -1.47321868896484e+02,
-9.84566025713160e-15, 1.00155668480983e-10, -4.53650557119545e-07,
1.35492335539311e-03, 5.20738172531128e+00, -2.93757507324219e+02,
-6.14934537701393e-16, 1.25114562807283e-11, -1.13348136210334e-07,
6.77172385621816e-04, 5.90138435363770e+00, -5.86281677246094e+02,
-3.84203159916016e-17, 1.56343210946930e-12, -2.83289747216031e-08,
3.38513898896053e-04, 6.59495878219604e+00, -1.17098315429688e+03,
-2.40086050186704e-18, 1.95397761556040e-13, -7.08123648607284e-09,
1.69238890521228e-04, 7.28832006454468e+00, -2.34003955078125e+03,
-1.50040998841287e-19, 2.44227686305946e-14, -1.77018322222722e-09,
8.46149268909357e-05, 7.98157405853271e+00, -4.67780566406250e+03,
-9.37716337755834e-21, 3.05272410607992e-15, -4.42530068145430e-10,
4.23063320340589e-05, 8.67477416992188e+00, -9.35299121093750e+03,
-5.86060190297109e-22, 3.81582889963464e-16, -1.10630546390489e-10,
2.11528840736719e-05, 9.36794853210449e+00, -1.87030156250000e+04,
-3.66283737740039e-23, 4.76973847894002e-17, -2.76573920016121e-11,
1.05763720057439e-05, 1.00611085891724e+01, -3.74027148437500e+04,
-2.28926113353121e-24, 5.96214332017297e-18, -6.91431720906133e-12,
5.28816826772527e-06, 1.07542629241943e+01, -7.48017734375000e+04,
-1.43078438741200e-25, 7.45266053865244e-19, -1.72857539913751e-12,
2.64407958638913e-06, 1.14474134445190e+01, -1.49599546875000e+05,
-8.94239086574533e-27, 9.31581404108818e-20, -4.32143388998454e-13,
1.32203877001302e-06, 1.21405620574951e+01, -2.99194718750000e+05,
-5.58899043923094e-28, 1.16447602812181e-20, -1.08035786263241e-13,
6.61019100789417e-07, 1.28337106704712e+01, -5.98384750000000e+05,
-3.49311782081312e-29, 1.45559453028129e-21, -2.70089380954809e-14,
3.30509465129580e-07, 1.35268573760986e+01, -1.19676450000000e+06,
-2.18319826185001e-30, 1.81949291041612e-22, -6.75223367683727e-15,
1.65254718353935e-07, 1.42200050354004e+01, -2.39352350000000e+06,
-1.36449879610682e-31, 2.27436598024797e-23, -1.68805831333020e-15,
8.26273591769677e-08, 1.49131526947021e+01, -4.78704150000000e+06,
-8.52811747566763e-33, 2.84295727809473e-24, -4.22014578332550e-16,
4.13136760357702e-08, 1.56062994003296e+01, -9.57407700000000e+06,
-5.33007296311479e-34, 3.55369659761841e-25, -1.05503637965693e-16,
2.06568380178851e-08, 1.62994461059570e+01, -1.91481460000000e+07,
-3.33129560194674e-35, 4.44212074702302e-26, -2.63759094914231e-17,
1.03284190089425e-08, 1.69925937652588e+01, -3.82962880000000e+07,
-2.08205975121671e-36, 5.55265093377877e-27, -6.59397737285579e-18,
5.16420950447127e-09, 1.76857414245605e+01, -7.65925680000000e+07,
-1.30128734451045e-37, 6.94081366722347e-28, -1.64849434321395e-18,
2.58210475223564e-09, 1.83788890838623e+01, -1.53185120000000e+08,
-8.13304660383952e-39, 8.67601708402933e-29, -4.12123585803487e-19,
1.29105237611782e-09, 1.90720348358154e+01, -3.06370240000000e+08,
-5.08315412739970e-40, 1.08450213550367e-29, -1.03030896450872e-19,
6.45526188058909e-10, 1.97651824951172e+01, -6.12740480000000e+08,
-3.17702387831723e-41, 1.35562766937958e-30, -2.57577241127179e-20,
3.22763094029455e-10, 2.04583301544189e+01, -1.22548096000000e+09
};

const float log_factorial_table[64] = {
0.0, 0.0, 0.6931471805599453, 1.791759469228055,
3.1780538303479458, 4.787491742782046, 6.579251212010101, 8.525161361065415,
10.60460290274525, 12.801827480081469, 15.104412573075516, 17.502307845873887,
19.987214495661885, 22.552163853123425, 25.19122118273868, 27.89927138384089,
30.671860106080672, 33.50507345013689, 36.39544520803305, 39.339884187199495,
42.335616460753485, 45.3801388984769, 48.47118135183522, 51.60667556776437,
54.78472939811232, 58.00360522298052, 61.261701761001994, 64.55753862700634,
67.88974313718154, 71.257038967168, 74.65823634883017, 78.0922235533153,
81.55795945611503, 85.05446701758152, 88.58082754219768, 92.13617560368708,
95.7196945421432, 99.33061245478743, 102.96819861451381, 106.63176026064346,
110.32063971475739, 114.03421178146169, 117.77188139974506, 121.53308151543864,
125.3172711493569, 129.12393363912722, 132.9525750356163, 136.80272263732635,
140.67392364823428, 144.56574394634487, 148.47776695177305, 152.40959258449735,
156.36083630307877, 160.3311282166309, 164.32011226319517, 168.32744544842765,
172.3527971391628, 176.39584840699735, 180.45629141754378, 184.53382886144948,
188.6281734236716, 192.73904728784487, 196.86618167288998, 201.00931639928152
};

const float lgamma_nu_func_approx_coeff3[] = {
1.15215471170606e+02, -7.81833294806814e+01,
2.17521294416039e+01, -4.00207969046817e+00,
1.73651454936872e+00, -4.64103368554928e+00,
4.96114611050933e+00, -2.62079988783550e+00,
2.19247677881418e-02, -2.31491872658262e-01,
9.80650507531970e-01, -1.33034436181266e+00,
2.59229669577681e-04, -1.11194941534111e-02,
1.96528569696233e-01, -3.30475312808754e-01,
3.59607301092398e-06, -6.23898142038213e-04,
4.51853154674520e-02, 4.54302652797463e-01,
5.43533760564972e-08, -3.78487592418411e-05,
1.10431898756058e-02, 1.17076361533367e+00,
8.42086542619799e-10, -2.34762247706483e-06,
2.74494910959480e-03, 1.86974602903652e+00,
1.31295320678586e-11, -1.46446291422155e-07,
6.85246648253346e-04, 2.56435212443809e+00,
2.05039288544523e-13, -9.14851640744505e-09,
1.71249747326460e-04, 3.25786403536974e+00,
3.20331056351859e-15, -5.71713887896017e-10,
4.28085671610666e-05, 3.95110239852806e+00,
5.00500544173177e-17, -3.57310494327578e-11,
1.07018999352172e-05, 4.64427237476749e+00,
7.82025560546465e-19, -2.23317388565474e-12,
2.67545986354366e-06, 5.33742525496218e+00,
1.22191238756488e-20, -1.39573107066068e-13,
6.68864021171837e-07, 6.03057386030030e+00,
1.90923698525687e-22, -8.72331476335181e-15,
1.67215943241429e-07, 6.72372140489075e+00,
2.98318428529838e-24, -5.45207321682807e-16,
4.18039894513541e-08, 7.41686859742537e+00,
4.66122520997206e-26, -3.40754560016380e-17,
1.04509969294582e-08, 8.11001581547534e+00,
7.28316485111643e-28, -2.12971617747160e-18,
2.61274939325342e-09, 8.80316297124324e+00,
1.13799534770463e-29, -1.33107347493960e-19,
6.53187647875674e-10, 9.49630979931091e+00
};

}   // namespace detail

template<class Alloc>
void get_log_stirling1_row(size_t n, std::vector<float, Alloc> & result) {
    result.resize(n + 1);
    detail::get_log_stirling1_row(n, result.data());
}

// --------------------------------------------------------------------------
// Explicit template instantiations

#define INSTANTIATE_TEMPLATES(Alloc)                \
    template void get_log_stirling1_row(            \
            size_t n,                               \
            std::vector<float, Alloc> & result);

INSTANTIATE_TEMPLATES(std::allocator<float>)
INSTANTIATE_TEMPLATES(aligned_allocator<float>)

#undef INSTANTIATE_TEMPLATES

}   // namespace distributions
