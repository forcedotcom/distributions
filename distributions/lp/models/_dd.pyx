ctypedef _h.Value Value


cdef class Shared:
    def __cinit__(self):
        self.ptr = new _h.Shared()

    def __dealloc__(self):
        del self.ptr


cdef class Group:
    def __cinit__(self):
        self.ptr = new _h.Group()

    def __dealloc__(self):
        del self.ptr

    def init(self, Shared shared):
        self.ptr.init(shared.ptr[0], get_rng()[0])

    def add_value(self, Shared shared, Value value):
        self.ptr.add_value(shared.ptr[0], value, get_rng()[0])

    def remove_value(self, Shared shared, Value value):
        self.ptr.remove_value(shared.ptr[0], value, get_rng()[0])

    def merge(self, Shared shared, Group source):
        self.ptr.merge(shared.ptr[0], source.ptr[0], get_rng()[0])


cdef class Mixture:
    def __cinit__(self):
        self.ptr = new _h.Mixture()

    def __dealloc__(self):
        del self.ptr

    def __len__(self):
        return self.ptr.groups.size()

    def __getitem__(self, int groupid):
        assert groupid < len(self), "groupid out of bounds"
        cdef Group group = Group()
        group.ptr[0] = self.ptr.groups[groupid]
        return group

    def append(self, Group group):
        self.ptr.groups.push_back(group.ptr[0])

    def clear(self):
        self.ptr.groups.clear()

    def init(self, Shared shared):
        self.ptr.init(shared.ptr[0], get_rng()[0])

    def add_group(self, Shared shared):
        self.ptr.add_group(shared.ptr[0], get_rng()[0])

    def remove_group(self, Shared shared, int groupid):
        self.ptr.remove_group(shared.ptr[0], groupid)

    def add_value(self, Shared shared, int groupid, Value value):
        self.ptr.add_value(shared.ptr[0], groupid, value, get_rng()[0])

    def remove_value(self, Shared shared, int groupid, Value value):
        self.ptr.remove_value(shared.ptr[0], groupid, value, get_rng()[0])

    def score_value(self, Shared shared, Value value,
              numpy.ndarray[numpy.float32_t, ndim=1] scores_accum):
        assert len(scores_accum) == self.ptr.groups.size(), \
            "scores_accum != len(mixture)"
        vector_float_from_ndarray(self.scores, scores_accum)
        self.ptr.score_value(shared.ptr[0], value, self.scores, get_rng()[0])
        vector_float_to_ndarray(self.scores, scores_accum)


def sample_value(Shared shared, Group group):
    cdef Value value = _h.sample_value(
        shared.ptr[0], group.ptr[0], get_rng()[0])
    return value


def sample_group(Shared shared, int size):
    cdef Group group = Group()
    cdef _h.Sampler sampler
    sampler.init(shared.ptr[0], group.ptr[0], get_rng()[0])
    cdef list result = []
    cdef int i
    cdef Value value
    for i in xrange(size):
        value = sampler.eval(shared.ptr[0], get_rng()[0])
        result.append(value)
    return result


def score_value(Shared shared, Group group, Value value):
    return _h.score_value(shared.ptr[0], group.ptr[0], value, get_rng()[0])


def score_group(Shared shared, Group group):
    return _h.score_group(shared.ptr[0], group.ptr[0], get_rng()[0])
