from nose.tools import assert_true

from distributions.models import dd
from distributions.models import dd_lp


def test_group_score():
    alphas = [0.2, 0.5, 1.0, 2.0]
    dim = len(alphas)

    model1 = dd.model_t()
    model2 = dd_lp.model_t()
    model1.dim = dim
    model2.dim = dim
    for i in range(dim):
        model1.alphas[i] = alphas[i]
        model2.alphas[i] = alphas[i]

    
