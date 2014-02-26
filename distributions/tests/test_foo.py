from nose.tools import assert_true

from distributions.models.dd import wrapped_foo as wf
from distributions.models.dd_lp import wrapped_foo as wf_lp


def test_foo():
    print 'wf', wf()
    print 'wf_lp', wf_lp()
