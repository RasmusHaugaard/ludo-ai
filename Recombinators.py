import numpy as np


class BaseRecombinator:
    name = "base"
    args = None

    def recombine(self, parent_a: np.ndarray, parent_b: np.ndarray):
        pass

    @staticmethod
    def new_gamma(size: int, alpha: float):
        return np.random.uniform(-alpha, 1 + alpha, size)

    def __call__(self, *args):
        return self.recombine(*args)


class NoneRecombinator(BaseRecombinator):
    name = "none"
    args = []

    def __init__(self, gene_count):
        pass

    def recombine(self, parent_a, parent_b):
        return parent_a, parent_b


class RealUniformRecombinator(BaseRecombinator):
    name = "uniform"
    args = []

    def __init__(self, gene_count):
        pass

    def recombine(self, parent_a, parent_b):
        mask = np.random.uniform(0, 1, parent_a.size) < 0.5
        child_1, child_2 = parent_a.copy(), parent_b.copy()
        child_1[mask] = parent_b[mask]
        child_2[mask] = parent_a[mask]
        return child_1, child_2


class RealWholeArithmeticRecombinator(BaseRecombinator):
    name = "whole"
    args = []

    def __init__(self, gene_count):
        self.blend_combinator = RealBlendRecombinator(gene_count, alpha=0)

    def recombine(self, *parents):
        return self.blend_combinator.recombine(*parents)


class RealBlendRecombinator(BaseRecombinator):
    name = "blend"
    args = [('alpha', float)]

    def __init__(self, gene_count, alpha):
        self.gene_count = gene_count
        self.alpha = alpha

    def recombine(self, parent_a, parent_b):
        gamma_1, gamma_2 = (self.new_gamma(parent_a.size, self.alpha) for _ in range(2))
        child_1 = gamma_1 * parent_a + (1 - gamma_1) * parent_b
        child_2 = gamma_2 * parent_b + (1 - gamma_2) * parent_a
        return child_1, child_2


def get_recombinator(name):
    recombinators = [NoneRecombinator, RealUniformRecombinator, RealWholeArithmeticRecombinator, RealBlendRecombinator]
    recombinator_map = {}
    for recombinator in recombinators:
        recombinator_map[recombinator.name] = recombinator
    return recombinator_map[name]
