try:
    from distributions import rng_cc
except ImportError:
    rng_cc = None

import numpy as np


class Rng:
    def __init__(self):
        print 'RNG_CC', rng_cc
        self.np = np
        if rng_cc is not None:
            self.cc = rng_cc.RngCc()
        else:
            self.cc = None


global_rng = Rng()
