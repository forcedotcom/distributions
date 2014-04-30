cimport _gp
import _gp

from distributions.mixins import GroupIoMixin, SharedIoMixin


NAME = 'GammaPoisson'
EXAMPLES = [
    {
        'shared': {'alpha': 1., 'inv_beta': 1.},
        'values': [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 2, 3],
    },
]
Value = int


cdef class _Shared(_gp.Shared):
    def load(self, raw):
        self.ptr.alpha = raw['alpha']
        self.ptr.inv_beta = raw['inv_beta']

    def dump(self):
        return {
            'alpha': self.ptr.alpha,
            'inv_beta': self.ptr.inv_beta,
        }


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(_gp.Group):
    def load(self, raw):
        self.ptr.count = raw['count']
        self.ptr.sum = raw['sum']
        self.ptr.log_prod = raw['log_prod']

    def dump(self):
        return {
            'count': self.ptr.count,
            'sum': self.ptr.sum,
            'log_prod': self.ptr.log_prod,
        }


class Group(_Group, GroupIoMixin):
    pass


Mixture = _gp.Mixture
sample_value = _gp.sample_value
sample_group = _gp.sample_group
score_group = _gp.score_group
