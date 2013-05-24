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

from util import discrete_draw_log, partition_from_counts
from math import factorial, log
from scipy.special import gammaln


"""
N is the total number of objects. K is the number of classes.

Implemented using Griffiths and Ghahramani, JMLR (2011), referred to below as
GG2011
"""


def draw_partition(x, alpha=1.):
    """
    Given a list of object ids, return a list of lists of ids representing a
    partition drawn from the CRP with the given alpha.
    """
    N = len(x)
    counts = draw_counts(N, alpha)
    return partition_from_counts(x, counts)


def draw_counts(N, alpha=1.):
    return draw(N, alpha)[0]


def draw_assignments(N, alpha=1.):
    return draw(N, alpha)[1]


def draw(N, alpha=1.0):
    hp = create_hp({'alpha': alpha})
    ss = create_ss()
    assignments = [0] * N
    for i in range(N):
        y = sample_data(hp, ss)
        add_data(ss, y)
        assignments[i] = y
    return dump_ss(ss)['counts'][:], assignments


def score(counts, alpha=1.):
    ss = create_ss(ss={'counts': counts})
    hp = create_hp(hp={'alpha': alpha})
    return data_prob(hp, ss)


def score_add(partition_size, collection_size, alpha):
    '''
    Returns differential score of adding one object
    to a partition of partition_size objects
    within a collection of collection_size objects.
    '''
    if partition_size == 0:
        return log(alpha / (collection_size + alpha))
    else:
        return log(partition_size / (collection_size + alpha))


def score_remove(partition_size, collection_size, alpha):
    '''
    Returns differential score of removing one object
    from a partition of partition_size objects
    within a collection of collection_size objects.
    '''
    if partition_size == 1:
        return -log(alpha / (collection_size - 1 + alpha))
    else:
        return -log((partition_size - 1) / (collection_size - 1 + alpha))


def create_ss(ss=None, p=None):
    if ss is None:
        ss = {}
    if p is None:
        p = {}
    counts = ss.get('counts', [])
    return {
        'counts': counts
        }


def dump_ss(ss):
    return {
        'counts': list(ss['counts'])
        }


def create_hp(hp=None, p=None):
    if hp is None:
        hp = {}
    alpha = hp.get('alpha', 1.)
    return {
        'alpha': alpha
        }


def dump_hp(hp):
    return {'alpha': hp['alpha']}


def add_data(ss, y):
    """
    Add an object to class y. Add the object to a new class if y == K.

    Raises a ValueError if y > K
    """
    K = len(ss['counts'])
    if y >= 0 and y < K:
        ss['counts'][y] += 1
    elif y == K:
        ss['counts'].append(1)
    else:
        raise ValueError("y is not valid: " + str(y))


def remove_data(ss, y):
    """
    Remove an object from class y.
    """
    K = len(ss['counts'])
    if y >= 0 and y < K:
        ss['counts'][y] -= 1
        ss['counts'] = [x for x in ss['counts'] if x > 0]
    else:
        raise ValueError("y is not valid: " + str(y))


def sample_data(hp, ss):
    K = len(ss['counts'])
    p = [pred_prob(hp, ss, y) for y in range(K + 1)]
    return discrete_draw_log(p)


def sample_post(hp, ss):
    raise NotImplementedError('CRP has no parameters to sample')


def pred_prob(hp, ss, y):
    """
    Returns the probability of assigning the next object to class y.
    If y == K, returns the probability of assigning the next object to a new
    class.

    Raises a ValueError if y > K

    GG2011, eq. 7
    """
    K = len(ss['counts'])
    N = sum(ss['counts'])
    if y >= 0 and y < K:
        return log(ss['counts'][y] / (N + hp['alpha']))
    elif y == K:
        return log(hp['alpha'] / (N + hp['alpha']))
    else:
        raise ValueError("y is not valid: " + str(y))


def data_prob(hp, ss):
    """
    Returns the log probability of the CRP configuration

    (log of) GG2011, eq. 5
    """
    K = len(ss['counts'])
    N = sum(ss['counts'])
    return K * log(hp['alpha']) \
        + sum(log(factorial(count - 1)) for count in ss['counts']) \
        + gammaln(hp['alpha']) - gammaln(hp['alpha'] + N)
