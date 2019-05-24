import numpy as np
import math


class BaseMutator:
    name = "base"
    args = []
    gene_count = None
    chromosome_length = None

    def mutate(self, chromosome: np.ndarray):
        pass

    def __call__(self, *args):
        return self.mutate(*args)


class NoneMutator(BaseMutator):
    name = "none"
    args = []

    def __init__(self, gene_count):
        self.gene_count = self.chromosome_length = gene_count
        pass

    def mutate(self, chromosome: np.ndarray):
        return chromosome


class RealNormalMutator(BaseMutator):
    name = "normal"
    args = [("sigma", float)]

    def __init__(self, gene_count: int, sigma: float):
        self.gene_count = self.chromosome_length = gene_count
        self.sigma = sigma
        pass

    def mutate(self, chromosome: np.ndarray):
        return chromosome + np.random.randn(self.gene_count) * self.sigma


class RealAdaptiveOneStepNormalMutator(BaseMutator):
    name = "one_step"
    args = [("lr", float)]

    def __init__(self, gene_count, lr):
        self.gene_count = gene_count
        self.chromosome_length = gene_count + 1
        self.tau = lr / math.sqrt(gene_count)
        self.sigma_min = 1e-3

    def mutate(self, chromosome: np.ndarray):
        genes, sigma = chromosome[:-1], chromosome[-1]
        sigma_ = max(self.sigma_min, sigma * pow(math.e, self.tau * np.random.randn()))
        genes_ = chromosome[:-1] + np.random.randn(chromosome.size - 1) * sigma_
        return np.concatenate((genes_, [sigma_]))


class RealAdaptiveNStepNormalMutator(BaseMutator):
    name = "n_step"
    args = [("lr_global", float), ("lr_local", float)]

    def __init__(self, gene_count, lr_global, lr_local):
        self.gene_count = gene_count
        self.chromosome_length = gene_count * 2
        self.tau_global = lr_global / math.sqrt(2 * gene_count)
        self.tau_local = lr_local / math.sqrt(2 * math.sqrt(gene_count))
        self.sigma_min = 1e-3

    def mutate(self, chromosome: np.ndarray):
        genes, sigma = chromosome[:self.gene_count], chromosome[self.gene_count:]
        sig_exp = self.tau_local * np.random.randn(self.gene_count) + self.tau_global * np.random.randn()
        sigma_ = np.maximum(self.sigma_min, sigma * np.exp(sig_exp))
        genes_ = genes + sigma_ * np.random.randn(self.gene_count)
        return np.concatenate((genes_, sigma_))


def get_mutator(name):
    mutators = [NoneMutator, RealNormalMutator, RealAdaptiveOneStepNormalMutator, RealAdaptiveNStepNormalMutator]
    mutator_map = {}
    for mutator in mutators:
        mutator_map[mutator.name] = mutator
    return mutator_map[name]
