try:
    import rng_cc
except ImportError:
    rng_cc = None

import numpy as np


class Rng:
    def __init__(self):
        self.np = np
        self.cc = rng_cc.RngCc()


global_rng = Rng()
