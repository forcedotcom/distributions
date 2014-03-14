# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from scipy.special import betaln, binom
from matplotlib import pyplot


def binomln(n, k):
    return np.log(binom(n, k))


def simone(alpha, beta, n, ITERS):
    ks = []
    for i in range(ITERS):
        theta = np.random.beta(alpha, beta)
        ks.append(np.random.binomial(n, theta))
    return np.array(ks)


def simtwo(alpha, beta, n, ITERS):
    counts = np.zeros((ITERS, 2), dtype=np.uint32)
    thetas = np.random.beta(alpha, beta, size=ITERS)
    for i in range(ITERS):
        if i % 1000000 == 0:
            print i

        counts[i] = np.random.binomial(n, thetas[i], size=2)
    return counts


class Hypers(object):

    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.N = 0

    def __str__(self):
        return "alpha=%f, beta=%f, N=%d" % (self.alpha,
                                            self.beta,
                                            self.N)


class SuffStats(object):

    def __init__(self):
        self.heads = 0
        self.tails = 0
        self.binomln_accum = 0
        self.dpcount = 0

    def __str__(self):
        return "heads=%d, tails=%d" % (self.heads, self.tails)


def add_dp(k, hps, suffs):
    suffs.heads += k
    suffs.tails += hps.N - k
    suffs.binomln_accum += binomln(hps.N, k)

    suffs.dpcount += 1


def compute_post_pred(k, hps, suffs):

    a = hps.alpha + suffs.heads
    b = hps.beta + suffs.tails

    return binomln(hps.N, k) + betaln(a + k, b + hps.N - k) - betaln(a, b)


def compute_total_likelihood(hps, suffs):

    a = hps.alpha + suffs.heads
    b = hps.beta + suffs.tails

    r = betaln(a, b) - betaln(hps.alpha, hps.beta)

    r += + suffs.binomln_accum

    return r


def two_d_hist(alpha, beta, N, ITERS):

    r = simtwo(alpha, beta, N, ITERS)
    h, b1, b2 = np.histogram2d(r[:, 0], r[:, 1], bins=N + 1)
    h[:, :] += 1
    pyplot.imshow(h)
    pyplot.show()

    logp = np.log(h.astype(float) / (ITERS))

    return logp


def test_post_pred_equal_likelihood():
    hps = Hypers()
    hps.N = 10

    for alpha in [5.0]:
        for beta in [5.0]:
            for ks in [[9, 1], [1, 9]]:
                print "-" * 60
                hps.alpha = alpha
                hps.beta = beta

                ss = SuffStats()

                pred_score = 0.0
                for ki, k in enumerate(ks):
                    pred_score += compute_post_pred(k, hps, ss)
                    add_dp(k, hps, ss)

                total_score = compute_total_likelihood(hps, ss)
                res = two_d_hist(alpha, beta, hps.N, 10000000)
                print res
                print res.shape
                emp_score = res[ks[0], ks[1]]
                d1 = abs(pred_score - emp_score)
                d2 = abs(emp_score - total_score)
                d3 = abs(total_score - pred_score)
                print "alpha=", alpha, "beta=", beta, "ks =", ks
                print pred_score, total_score, emp_score
                print "error=", d1 + d2 + d3
