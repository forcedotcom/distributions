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
from distributions.mixins import Serializable, ProtobufSerializable


class LowEntropy(Serializable, ProtobufSerializable):

    def __init__(self, dataset_size=0):
        self.dataset_size = int(dataset_size)

    def load(self, raw):
        self.dataset_size = int(raw['dataset_size'])
        assert self.dataset_size >= 0

    def dump(self):
        return {'dataset_size': self.dataset_size}

    def load_protobuf(self, message):
        self.dataset_size = int(message.dataset_size)

    def dump_protobuf(self, message):
        message.Clear()
        message.dataset_size = self.dataset_size

    def sample_assignments(self, sample_size):
        assert sample_size <= self.dataset_size

        assignments = []
        counts = []
        scores = []
        size = 1
        unused = 0

        for _ in xrange(sample_size):

            score_empty = self.score_add_value(0, unused, size)
            if len(counts) == 0 or counts[-1] != 0:
                counts.append(0)
                scores.append(score_empty)
            else:
                scores[-1] = score_empty

            assign = sample_discrete_log(scores)
            counts[assign] += 1
            size += 1
            scores[assign] = self.score_add_value(counts[assign], unused, size)
            assignments.append(assign)

        return assignments

    def score_counts(self, counts):
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
            score += self._approximate_dataprob_correction(
                sample_size,
                self.dataset_size)
        score -= self._cluster_normalizing_score(sample_size)
        return score

    def score_add_value(
            self,
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count=1):
        # see `python derivations/clustering.py fastlog`
        very_large = 10000

        if group_size == 0:
            if sample_size == self.dataset_size:
                return 0.0
            else:
                return (self._approximate_postpred_correction(sample_size)
                        - log(empty_group_count))
        elif group_size > very_large:
            bigger = 1.0 + group_size
            return 1.0 + log(bigger)
        else:
            bigger = 1.0 + group_size
            return log(bigger / group_size) * group_size + log(bigger)

    def score_remove_value(
            self,
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count=1):
        group_size -= 1
        return -self.score_add_value(
            group_size,
            nonempty_group_count,
            sample_size,
            empty_group_count)

    def _approximate_postpred_correction(self, sample_size):
        '''
        ad hoc approximation,
        see `python derivations/clustering.py postpred`
        see `python derivations/clustering.py approximations`
        '''
        exponent = 0.45 - 0.1 / sample_size - 0.1 / self.dataset_size
        scale = self.dataset_size / sample_size
        return log(scale) * exponent

    # this code was generated by derivations/clustering.py
    _cluster_normalizing_scores = [
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

    # this code was generated by derivations/clustering.py
    @staticmethod
    def _cluster_normalizing_score(sample_size):
        # TODO incorporate dataset_size for higher accuracy
        n = sample_size
        if n < 48:
            return LowEntropy._cluster_normalizing_scores[n]
        else:
            coeff = 0.28269584
            log_z_max = n * log(n)
            return log_z_max * (1.0 + coeff * n ** -0.75)

    @staticmethod
    def _approximate_dataprob_correction(sample_size, dataset_size):
        '''
        ad hoc approximation,
        see `python derivations/clustering.py dataprob`
        see `python derivations/clustering.py approximations`
        '''
        n = log(sample_size)
        N = log(dataset_size)
        return 0.061 * n * (n - N) * (n + N) ** 0.75

    #-------------------------------------------------------------------------
    # Examples

    EXAMPLES = [
        {'dataset_size': 5},
        {'dataset_size': 10},
        {'dataset_size': 100},
        {'dataset_size': 1000},
    ]
