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

import math
from collections import defaultdict
from nose import SkipTest
from nose.tools import assert_less, assert_greater
from distributions.util import discrete_goodness_of_fit
from distributions.tests.util import seed_all
try:
    from distributions.lp.clustering import (
        count_assignments,
        PitmanYor,
        LowEntropy,
    )
except ImportError:
    raise SkipTest('no cython support')


MODELS = [
    PitmanYor,
    LowEntropy,
]

SAMPLE_COUNT = 1000
MAX_SIZE = 5
MIN_GOODNESS_OF_FIT = 1e-3


def canonicalize(assignments):
    groups = defaultdict(lambda: [])
    for value, group in enumerate(assignments):
        groups[group].append(value)
    result = []
    for group in groups.itervalues():
        group.sort()
        result.append(tuple(group))
    result.sort()
    return tuple(result)


def test_models(Model=None, size=None):
    seed_all(0)
    for Model in MODELS:
        for size in xrange(2, MAX_SIZE + 1):
            yield _test_models, Model, size


def _test_models(Model, size):
        model = Model()

        if Model.__name__ == 'LowEntropy':
            raise SkipTest('FIXME LowEntropy.score_counts is not normalized')

        for i, EXAMPLE in enumerate(Model.EXAMPLES):
            print 'Example {}'.format(i)
            model.load(EXAMPLE)
            samples = []
            probs_dict = {}
            for _ in xrange(SAMPLE_COUNT):
                value = model.sample_assignments(size)
                assignments = dict(enumerate(value))
                counts = count_assignments(assignments)
                prob = math.exp(model.score_counts(counts))
                sample = canonicalize(value)
                samples.append(sample)
                probs_dict[sample] = prob

            total = sum(probs_dict.values())
            assert_less(
                abs(total - 1),
                1e-2,
                'not normalized: {}'.format(total))

            gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
            print '{} gof = {:0.3g}'.format(Model.__name__, gof)
            assert_greater(gof, MIN_GOODNESS_OF_FIT)
