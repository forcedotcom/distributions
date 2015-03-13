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

from distributions.dbg.special import log
from distributions.dbg.random import sample_discrete_log
from distributions.mixins import SharedIoMixin


class LowEntropy(SharedIoMixin):
    '''
    A parameter-free clustering prior based on partition entropy.

    See /derivations/clustering.py for intro and motivation.

    Definition
    ----------

    Let X = [X_0,...,X_{N-1}] be an assignment vector of N points into
    clusters of sizes [N_0,...,N_{K-1}], i.e.

        N_k = #{i | X_i = k, i in {0,...,N-1}}

    The empirical entropy of the assignment vector is

        H(X) = sum_k -p_k log p_k  = sum_k N_k/N log( N_k/N )

    And so the information complexity of the assignment vector is N H(X),
    where each assignment requires H(X) information to specify.
    Define the prior probability in terms of cluster sizes [N_1,...,N_k] as

        P(X)   propto   exp(-N H(X))   propto   prod_{k=1}^K N_k^{N_k}

    Implementation
    --------------

    This class implements approximations to numerical functions
    required in MCMC inference using this clustering prior:

    - evaluating log posterior predictive probability of an assignment
      given a partial assignment vector

        score_add_value(...) =

            log P[ X_i=x_i | X_0=x_0,...,X_{i-1}=x_{i-1} ] Z / Z_approx

    - evaluating log probability of a full assignment vector,
      up to the constant partition function

        score_counts(...) =

            log P[ X_0=x_0,...,X_{n-1}=x_{n-1} ] Z(n) / Z_approx(n)

    - sampling a full assignment vector X ~ P[ X=x ],

        X = model.sample_assignments(...)

    - evaluating the partition function Z(n) exactly for small n
      and at low accuracy for large n

        log_partition_function(n) = log Z_exact(n)  for n <= 47
                                    log Z_approx(n) for n > 47
    '''

    def __init__(self, dataset_size=0):
        self.dataset_size = int(dataset_size)

    # ------------------------------------------------------------------------
    # Serialization

    def load(self, raw):
        self.dataset_size = int(raw['dataset_size'])
        assert self.dataset_size >= 0

    def dump(self):
        return {'dataset_size': self.dataset_size}

    def protobuf_load(self, message):
        self.dataset_size = int(message.dataset_size)

    def protobuf_dump(self, message):
        message.Clear()
        message.dataset_size = self.dataset_size

    # ------------------------------------------------------------------------
    # Sampling

    def sample_assignments(self, sample_size):
        '''
        Sample partial assignment vector

            [X_0, ..., X_{n-1}]

        where

            n = sample_size <= N = dataset_size.
        '''
        assert sample_size <= self.dataset_size

        assignments = []
        counts = []
        scores = []
        bogus = 0

        for size in xrange(sample_size):

            score_empty = self.score_add_value(0, bogus, size)
            if len(counts) == 0 or counts[-1] != 0:
                counts.append(0)
                scores.append(score_empty)
            else:
                scores[-1] = score_empty

            assign = sample_discrete_log(scores)
            counts[assign] += 1
            size += 1
            scores[assign] = self.score_add_value(counts[assign], bogus, bogus)
            assignments.append(assign)

        return assignments

    # ------------------------------------------------------------------------
    # Scoring

    def score_counts(self, counts):
        '''
        Return log probability of data, given sufficient statistics of
        a partial assignment vector [X_0,...,X_{n-1}]

            log P[ X_0=x_0, ..., X_{n-1}=x_{n-1} ]
        '''
        score = 0.0
        sample_size = 0
        for count in counts:
            sample_size += count
            if count > 1:
                score += count * log(count)
        assert sample_size <= self.dataset_size

        if sample_size != self.dataset_size:
            log_factor = self._approximate_postpred_correction(sample_size)
            score += log_factor * (len(counts) - 1)
            score += self._approximate_dataprob_correction(sample_size)
        score -= self.log_partition_function(sample_size)
        return score

    def score_add_value(
            self,
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count=1):
        '''
        Return log of posterior predictive probability given
        sufficient statistics of a partial assignments vector [X_0,...,X_{n-1}]

            log P[ X_n = k | X_0=x_0, ..., X_{n-1}=x_{n-1} ]

        where

            group_size = #{i | x_i = k, i in {0,...,n-1}}

            nonempty_group_count = #{x_i | i in {0,...,n-1}}

            sample_size = n

        and empty_group_count is the number of empty groups that are uniformly
        competing for the assignment.  Typically empty_group_count = 1, but
        multiple empty "ephemeral" groups are used in e.g. Radford Neal's
        Algorithm-8 \cite{neal2000markov}.
        '''
        assert sample_size < self.dataset_size
        assert 0 < empty_group_count

        if group_size == 0:
            score = -log(empty_group_count)
            if sample_size + 1 < self.dataset_size:
                score += self._approximate_postpred_correction(sample_size + 1)
            return score

        # see `python derivations/clustering.py fastlog`
        very_large = 10000
        bigger = 1.0 + group_size
        if group_size > very_large:
            return 1.0 + log(bigger)
        else:
            return log(bigger / group_size) * group_size + log(bigger)

    def score_remove_value(
            self,
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count=1):
        '''
        Reverse transition probability of score_add_value, given
        sufficient statistics of the partial assignment vector  [X_0,...,X_n}]

            -log P[ X_n=x_n | X_0=x_0, ..., X_{n-1}=x_{n-1} ]

        This is useful in Metropolis-Hastings inference.
        '''
        assert sample_size > 0

        group_size -= 1
        return -self.score_add_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count)

    # ------------------------------------------------------------------------
    # Approximations

    # this code was generated by derivations/clustering.py
    def log_partition_function(self, sample_size):
        '''
        Computes

            log_sum_exp(
                sum(n * log(n) for n in partition)
                for partition in partitions(sample_size)
            )

        exactly for small n, and approximately for large n.
        '''
        # TODO incorporate dataset_size for higher accuracy
        n = sample_size
        if n < 48:
            return LowEntropy.log_partition_function_table[n]
        else:
            coeff = 0.28269584
            log_z_max = n * log(n)
            return log_z_max * (1.0 + coeff * n ** -0.75)

    # this code was generated by derivations/clustering.py
    log_partition_function_table = [
        0.00000000, 0.00000000, 1.60943791, 3.68887945, 6.07993320,
        8.70549682, 11.51947398, 14.49108422, 17.59827611, 20.82445752,
        24.15668300, 27.58456586, 31.09958507, 34.69462231, 38.36364086,
        42.10145572, 45.90356476, 49.76602176, 53.68533918, 57.65841234,
        61.68245958, 65.75497413, 69.87368527, 74.03652635, 78.24160846,
        82.48719834, 86.77169993, 91.09363859, 95.45164780, 99.84445762,
        104.27088480, 108.72982416, 113.22024112, 117.74116515, 122.29168392,
        126.87093829, 131.47811772, 136.11245629, 140.77322911, 145.45974907,
        150.17136399, 154.90745399, 159.66742919, 164.45072752, 169.25681285,
        174.08517319, 178.93531914, 183.80678238
    ]

    def _approximate_postpred_correction(self, sample_size):
        '''
        ad hoc approximation,
        see `python derivations/clustering.py postpred`
        see `python derivations/clustering.py approximations`
        '''
        assert 0 < sample_size
        assert sample_size < self.dataset_size

        exponent = 0.45 - 0.1 / sample_size - 0.1 / self.dataset_size
        scale = self.dataset_size / sample_size
        return log(scale) * exponent

    def _approximate_dataprob_correction(self, sample_size):
        '''
        ad hoc approximation,
        see `python derivations/clustering.py dataprob`
        see `python derivations/clustering.py approximations`
        '''
        n = log(sample_size)
        N = log(self.dataset_size)
        return 0.061 * n * (n - N) * (n + N) ** 0.75

    # ------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {'dataset_size': 5},
        {'dataset_size': 1000},
    ]
