cimport _nich
import _nich

from distributions.mixins import GroupIoMixin, SharedIoMixin


NAME = 'NormalInverseChiSq'
EXAMPLES = [
    {
        'shared': {'mu': 0., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
        'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    },
]
Value = float


cdef class _Shared(_nich.Shared):
    def load(self, dict raw):
        self.ptr.mu = raw['mu']
        self.ptr.kappa = raw['kappa']
        self.ptr.sigmasq = raw['sigmasq']
        self.ptr.nu = raw['nu']

    def dump(self):
        return {
            'mu': self.ptr.mu,
            'kappa': self.ptr.kappa,
            'sigmasq': self.ptr.sigmasq,
            'nu': self.ptr.nu,
        }

    def load_protobuf(self, message):
        self.ptr.mu = float(message.mu)
        self.ptr.kappa = float(message.kappa)
        self.ptr.sigmasq = float(message.sigmasq)
        self.ptr.nu = float(message.nu)

    def dump_protobuf(self, message):
        message.Clear()
        message.mu = self.ptr.mu
        message.kappa = self.ptr.kappa
        message.sigmasq = self.ptr.sigmasq
        message.nu = self.ptr.nu


class Shared(_Shared, SharedIoMixin):
    pass


cdef class _Group(_nich.Group):
    def load(self, dict raw):
        self.ptr.count = raw['count']
        self.ptr.mean = raw['mean']
        self.ptr.count_times_variance = raw['count_times_variance']

    def dump(self):
        return {
            'count': self.ptr.count,
            'mean': self.ptr.mean,
            'count_times_variance': self.ptr.count_times_variance,
        }


class Group(_Group, GroupIoMixin):
    pass


class Sampler(_nich.Sampler):
    pass


Mixture = _nich.Mixture
sample_value = _nich.sample_value
sample_group = _nich.sample_group
score_group = _nich.score_group
