import numpy as np
from scipy.special import betaln, gammaln
from matplotlib import pyplot
from loggamma import gammaln_approx_second_order


def test_betaln(x, y):
    return (gammaln_approx_second_order(x)
            + gammaln_approx_second_order(y)
            - gammaln_approx_second_order(x + y))


def test_pp():
    N = 20

    ms = np.linspace(1, 1e1, N)
    ns = np.linspace(1, 1e1, N)
    alphas = np.linspace(1, 1e1, N)
    betas = np.linspace(1, 1e1, N)

    TERMS = 17

    A = np.zeros((N * N * N * N, TERMS))
    B = np.zeros(N * N * N * N)

    pos = 0
    for mi, m in enumerate(ms):
        print mi
        for ni, n in enumerate(ns):
            for ai, a in enumerate(alphas):
                for bi, b in enumerate(betas):

                    A[pos, :] = (1, m, n, a, b,
                                 m * n, m * a, n * a, m * b, n * b,
                                 np.log(m + a),
                                 np.log(n + b), np.log(m + n + a + b),
                                 np.log(a), np.log(b),
                                 a * np.log(a), b * np.log(b))

                    B[pos] = test_betaln(m + a, n + b) - test_betaln(a, b)

                    pos += 1

    x, residues, rank, s = np.linalg.lstsq(A, B)
    print x
    print residues

    pyplot.plot(((np.dot(A, x) - B) / B) * 100)
    pyplot.ylabel("Percent error")

    pyplot.show()


def test_pp2():
    for MAX in [1e1, 1e2, 1e3, 1e5, 1e7]:
        N = 20
        ms = np.linspace(1, MAX, N)
        ns = np.linspace(1, MAX, N)
        alphas = np.linspace(1, MAX, N)
        betas = np.linspace(1, MAX, N)

        A = np.zeros(N * N * N * N)
        B = np.zeros(N * N * N * N)

        pos = 0
        for mi, m in enumerate(ms):
            print mi
            for ni, n in enumerate(ns):
                for ai, a in enumerate(alphas):
                    for bi, b in enumerate(betas):

                        B[pos] = betaln(m + a, n + b) - betaln(a, b)
                        A[pos] = test_betaln(m + a, n + b) - test_betaln(a, b)

                        pos += 1

        pyplot.figure()
        pyplot.plot(((A - B) / B) * 100)
        pyplot.ylabel("Percent error")
        pyplot.title("MAX=%f" % MAX)

    pyplot.show()


def test2():
    N = 1000

    x = np.linspace(1, 1e1, N)

    y = gammaln(x)
    u = x - 0.5
    yhat = np.log(2 * 3.1415) / 2. + u * np.log(u) - u
    pyplot.plot(x, y)
    pyplot.plot(x, yhat)
    pyplot.show()
