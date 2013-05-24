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
from math import log


"""
A Pitman-Yor Process that supports the sequential generation of a partition.

Implemented using Teh, "A hierarchical Bayesian language model based on
Pitman-Yor Processes", ACL (2006)

We use parameter names d and alpha, and
c_k => counts[k]
t   => K
c_. => N

Also note that successive draws from G_0 are just the integers.

According to Teh, "There is in general no known analytic form for the density
of PY(d, alpha, G_0) [in the finite case]", so there is not method that
provides the total log probability of the current PY configuration (cf. the
CRP, which does have this method).

Note that for d == 0, the Pitman-Yor process reduces to the CRP

A more complete account of the PYP is given in Pitman,
"Combinatorial Stochastic Processes" UCB TR (2002), p. 62ff.
"""


def draw_partition(x, alpha=1., d=0.):
    """
    Given a list of object ids, return a list of lists of ids representing a
    partition drawn from the PYP with the given d and alpha.
    """
    N = len(x)
    counts = draw_counts(N, alpha, d)
    return partition_from_counts(x, counts)


def draw_counts(N, alpha=1., d=0.):
    return draw(N, alpha, d)[0]


def draw_assignments(N, alpha=1., d=0.):
    return draw(N, alpha, d)[1]


def draw(N, alpha=1.0, d=0.):
    hp = create_hp({'alpha': alpha, 'd': d})
    ss = create_ss()
    assignments = [0] * N
    for i in range(N):
        y = sample_data(hp, ss)
        add_data(ss, y)
        assignments[i] = y
    return dump_ss(ss)['counts'][:], assignments


def score(counts, alpha, d):
    ss = create_ss(ss={'counts': counts})
    hp = create_hp(hp={'alpha': alpha, 'd': d})
    return data_prob(hp, ss)


def score_add(partition_size, K, N, alpha, d):
    '''
    Returns differential score of adding one object
    to a partition of partition_size objects
    within a collection of N objects in K partitions.

    Note that, unlike the CRP, this depends on K
    '''
    if partition_size == 0:
        return log((alpha + d * K) / (alpha + N))
    else:
        return log((partition_size - d) / (alpha + N))


def score_remove(partition_size, K, N, alpha, d):
    '''
    Returns differential score of removing one object
    from a partition of partition_size objects
    within a collection of N objects in K partitions.

    Note that, unlike the CRP, this depends on K
    '''
    if partition_size == 1:
        return -log((alpha + d * (K - 1)) / (alpha + N - 1))
    else:
        return -log((partition_size - 1 - d) / (alpha + N - 1))


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
    d = hp.get('d', 0.)
    return {
        'alpha': alpha,
        'd': d
        }


def dump_hp(hp):
    return {'alpha': hp['alpha'], 'd': hp['d']}


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
        raise ValueError("k is not valid: " + str(y))


def remove_data(ss, y):
    """
    Remove an object from class y.
    """
    K = len(ss['counts'])
    if y >= 0 and y < K:
        ss['counts'][y] -= 1
        ss['counts'] = [x for x in ss['counts'] if x > 0]
    else:
        raise ValueError("k is not valid: " + str(y))


def sample_data(hp, ss):
    K = len(ss['counts'])
    p = [pred_prob(hp, ss, y) for y in range(K + 1)]
    return discrete_draw_log(p)


def sample_post(hp, ss):
    raise NotImplementedError('PYP has no parameters to sample')


def pred_prob(hp, ss, y):
    """
    Returns the probability of assigning the next object to class k.
    If k == K, returns the probability of assigning the next object to a new
    class.

    Raises a ValueError if k > K

    GG2011, eq. 7
    """
    K = len(ss['counts'])
    N = sum(ss['counts'])
    assert y >= 0 and y <= K
    if y < K:
        return log((ss['counts'][y] - hp['d']) / (hp['alpha'] + N))
    elif y == K:
        return log((hp['alpha'] + hp['d'] * K) / (hp['alpha'] + N))


def data_prob(hp, ss):
    lp = 0.
    ss_build = create_ss()
    for i, group_size in enumerate(ss['counts']):
        if group_size == 0:
            ss_build['counts'].append(0)
        else:
            for _ in range(group_size):
                lp += pred_prob(hp, ss_build, i)
                if i == len(ss_build['counts']):
                    ss_build['counts'].append(1)
                else:
                    ss_build['counts'][i] += 1
    return lp
