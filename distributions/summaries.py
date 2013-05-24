# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import math
from collections import Counter

from distributions.names import CANONICAL


def summarize_continuous(data):
    """
    Returns a dict that summarizes the specified data, which must
    be a list-like object containing continuous values.
    """
    summary = {}
    summary['type'] = 'continuous'
    summary['min'] = np.nanmin(data)
    summary['max'] = np.nanmax(data)
    summary['n'] = len(data)
    summary['sum'] = np.sum(data)
    summary['mean'] = np.mean(data)
    summary['median'] = np.median(data)
    summary['std'] = np.std(data)
    summary['var'] = np.var(data)
    s = sorted(data)
    summary['deciles'] = [s[int(math.floor((i + 1) * .1 * len(s)))]
        for i in range(9)]
    summary['quartiles'] = [s[int(math.floor(((i + 1) * .25 * len(s))))]
        for i in range(3)]
    summary['percentiles'] = [s[int(math.floor((i + 1) * .01 * len(s)))]
        for i in range(99)]

    return summary


def summarize_categorical(data):
    """
    Returns a dict that summarizes the specified data, which must
    be a list-like object containing category ids as 0-indexed ints

    Note that the mode returned is just the first mode found; there may
    be others that can be identified by examining the counts.
    """
    summary = {}
    summary['type'] = 'categorical'
    summary['dim'] = np.max(data) + 1
    summary['n'] = len(data)
    counts = [0] * summary['dim']
    for x in data:
        counts[x] += 1
    summary['counts'] = counts
    summary['frequencies'] = [x / float(summary['n']) for x in counts]
    summary['mode'] = counts.index(np.max(counts))
    return summary


def summarize_unbounded_categorical(data):
    """
    Returns a dict that summarizes the specified data, which must
    be a list-like object.

    Note that the mode returned is just the first mode found; there may
    be others that can be identified by examining the counts.
    """
    summary = {}
    summary['type'] = 'unbounded_categorical'
    summary['dim'] = len(np.unique(data))
    summary['n'] = len(data)
    counts = Counter(data)
    summary['counts'] = counts
    summary['frequencies'] = {key: count / summary['n']
            for key, count in counts.items()}
    summary['mode'] = counts.most_common(1)
    return summary


def summarize_discrete(data):
    """
    Returns a dict that summarizes the specified data, which must
    be a list-like object containing integers.
    """
    raise NotImplementedError


NAMES = {
    'dd': summarize_categorical,
    'dpm': summarize_unbounded_categorical,
    'bb': summarize_categorical,
    'gp': summarize_continuous,
    'nich': summarize_continuous,
    'ngig': summarize_continuous,
    }


def summarize(name, data):
    name = name.lower()
    name = CANONICAL.get(name, name)
    func = NAMES[name]
    return func(data)
