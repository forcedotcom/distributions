import numpy
from nose.tools import assert_equal
from distributions.lp.special import log_stirling1_row
from distributions.tests.util import assert_close


def test_log_stirling1_row():
    MAX_N = 128

    rows = [[1]]
    for n in range(1, MAX_N + 1):
        prev = rows[-1]
        row = [0] + [(n-1) * prev[k] + prev[k-1] for k in range(1, n)] + [1]
        rows.append(row)

    for n in range(1, MAX_N + 1):
        print 'Row {}:'.format(n),
        row_py = numpy.log(numpy.array(rows[n][1:], dtype=numpy.double))
        row_cpp = log_stirling1_row(n)[1:]
        assert_equal(len(row_py), len(row_cpp))

        # only the slopes need to be accurate
        #print 0,
        #assert_close(row_py[0], row_cpp[0])
        #print len(row_py)
        #assert_close(row_py[-1], row_cpp[-1])

        diff_py = numpy.diff(row_py)
        diff_cpp = numpy.diff(row_cpp)
        for k_minus_1, (dx_py, dx_cpp) in enumerate(zip(diff_py, diff_cpp)):
            k = k_minus_1 + 1
            print '%d-%d' % (k, k + 1),
            assert_close(dx_py, dx_cpp, tol=0.5)
        print
