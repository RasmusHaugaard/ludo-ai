import argparse

import numpy as np
from matplotlib import pyplot as plt

from ga_utils import load_populations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()

    generation_ids, genes, sigmas = load_populations(args.path)
    flat_genes = genes.reshape((-1, genes.shape[-1]))

    def plot(gene_idx):
        plt.figure('{}/{}'.format(*gene_idx), figsize=(3, 3))
        c = np.arange(len(flat_genes))

        all_genes = flat_genes[:, gene_idx]

        xs = all_genes[:, 0]
        ys = all_genes[:, 1]
        plt.scatter(xs, ys, c=c, cmap="brg", s=8)

        ax = plt.gca()
        ax.tick_params(length=0)
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        plt.grid()
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.tight_layout(pad=0)

    plot((0, 1))
    plot((2, 3))
    plt.show()


if __name__ == '__main__':
    main()
