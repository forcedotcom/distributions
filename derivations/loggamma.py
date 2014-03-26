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
from scipy.special import gammaln, betaln
from matplotlib import pyplot


def create_coeff(N, order):
    points = 2. ** np.arange(0, N + 1)
    coeff = np.zeros((N, order + 1), dtype=np.float32)

    for pos in range(N):
        left = points[pos]
        right = points[pos + 1]

        xrange = np.linspace(left, right, 100)

        y = gammaln(xrange)
        z = np.polyfit(xrange, y, order)
        print z.shape
        coeff[pos] = z
    return coeff

coeff = create_coeff(33, 2)
coeff3 = create_coeff(33, 3)
coeff4 = create_coeff(33, 5)
coeff5 = create_coeff(33, 5)


def gammaln_approx_second_order(x):
    if x < 2.5:
        return gammaln(x)

    pos = int(np.floor(np.log2(x)))

    a = coeff[pos]

    yhat = a[2] + a[1] * x + a[0] * x ** 2

    return yhat


def gammaln_approx_third_order(x):
    if x < 2.5:
        return gammaln(x)

    pos = int(np.floor(np.log2(x)))

    a = coeff3[pos]

    yhat = a[3] + a[2] * x + a[1] * x ** 2 + a[0] * x ** 3

    return yhat


def gammaln_approx_fourth_order(x):
    if x < 2.5:
        return gammaln(x)

    pos = int(np.floor(np.log2(x)))

    a = coeff4[pos]

    yhat = a[4] + a[3] * x + a[2] * x ** 2 + a[1] * x ** 3 + a[0] * x ** 4

    return yhat


def gammaln_approx_fifth_order(x):
    # if x < 2.5:
    # return gammaln(x)

    pos = int(np.floor(np.log2(x)))

    a = coeff5[pos]

    yhat = a[5] + a[4] * x + a[3] * x ** 2 + \
        a[2] * x ** 3 + a[1] * x ** 4 + a[0] * x ** 5

    return yhat


def func_test(approx_func, max):
    #x = np.array([1000.0, 1001, 1002])
    x = np.linspace(0.001, max, 10000)

    y = gammaln(x)
    yhat = map(approx_func, x)

    pyplot.subplot(2, 1, 1)
    pyplot.plot(x, y, linewidth=2)
    pyplot.plot(x, yhat, color='r')
    pyplot.subplot(2, 1, 2)

    delta = yhat - y
    err_frac = (delta / gammaln(x))
    pyplot.plot(x, err_frac * 100)
    THOLD = 0.001
    accuracy = np.sum(np.abs(err_frac) < THOLD).astype(float) / len(x) * 100
    print "accurate", accuracy
    pyplot.ylabel('percent error')

    pyplot.show()


def test_fifth_order():
    func_test(gammaln_approx_fifth_order, 1e2)


def test_gauss():
    func_test(gammaln_gauss, 1e2)


def beta(alpha, beta):
    gl = gammaln_approx_second_order
    return gl(alpha) + gl(beta) - gl(alpha + beta)


def gammaln_gauss(x):
    if x < 2.5:
        return gammaln(x)
    u = x - 0.5
    return np.log(2 * 3141592) / 2 + u * np.log(u) - u


def test_beta():
    N = 100
    xs = np.linspace(0.01, 1e5, N)
    ys = np.linspace(0.01, 1e5, N)

    vals = np.zeros((N, N))
    errs = np.zeros(N * N)

    pos = 0
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            z = betaln(x, y)
            zhat = beta(x, y)
            errs[pos] = (zhat - z) / z
            vals[xi, yi] = z
            pos += 1

    pyplot.figure()
    pyplot.plot(errs)
    # pyplot.imshow(vals)
    # pyplot.figure()
    ## PN = 5
    # for i in range(PN):
    ##     pyplot.subplot(1, PN, i+1)
    ##     pyplot.plot(ys, vals[N/ PN * i, :])
    ##     pyplot.title(xs[N/PN * i])
#    pyplot.plot(delta)
    pyplot.show()


def coeff_gen():
    mycoeff = create_coeff(33, 5)
    print "const float coeff[] = {",
    for a in mycoeff:
        for ai in a:
            print "%.14e," % ai,
        print
    print "};"


def lt25test():
    x = np.linspace(0.001, 2.5, 1000)
    y = gammaln(x)
    order = 3

    z = np.polyfit(x, y, order)
    w = np.poly1d(z)
    pyplot.plot(x, y)
    pyplot.plot(x, w(x))
    pyplot.figure()
    delta = np.abs(y - w(x))
    print delta
    pyplot.plot(x, delta / y * 100)
    pyplot.ylabel("percent error")
    pyplot.grid(1)
    pyplot.show()


def lstudent():

    def tgtfunc(x):
        return gammaln(x / 2 + 0.5) - gammaln(x / 2)

    coeffs = []
    pot_range = 2
    for pot in np.arange(-4, 32, pot_range):

        x = np.linspace(2.0 ** pot, 2.0 ** (pot + pot_range), 1000)
        y = tgtfunc(x)

        order = 3
        z = np.polyfit(x, y, order)
        coeffs.append(z)

        print z

        ## w = np.poly1d(z)
        ## yhat = w(x)
        ## pyplot.plot(x, y)
        ## pyplot.plot(x, yhat)

        # pyplot.show()
    print "const float lgamma_nu_func_approx_coeff3[] = {",
    for a in coeffs:
        for ai in a:
            print "%.14e," % ai,
        print
    print "};"
