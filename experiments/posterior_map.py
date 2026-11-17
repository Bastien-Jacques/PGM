"""True posteriors of several images on the same latent plane, to choose pairs for the morphing."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from grid import make_grid, posterior_on_grid
from utils import load_vae, load_dataset

# settings
vae = "vae_ffg_h100_tanh_seed0"
input_dim, hidden_dims, latent_dim, act = 784, [100], 2, "tanh"
L, n = 8.0, 300                     # global grid covering the region where posteriors live
level_fractions = [0.1, 0.5]        # contour levels, as fractions of each posterior's maximum


@torch.no_grad()
def posteriors(X, dec):
    """Exact posteriors of a batch of images on the global grid, and their means."""
    Z, lp, lda = make_grid(L, n)
    post = posterior_on_grid(X, Z, dec, lp, lda)                 # (B, n * n)
    means = (post * math.exp(lda)) @ Z                           # (B, 2) posterior means
    g = torch.linspace(-L, L, n)
    return post, means, g


def draw(ax, g, dens, color):
    """Contour lines of one posterior, in a single colour."""
    D = dens.view(n, n).T.numpy()                                # contour expects (y, x) indexing
    ax.contour(g.numpy(), g.numpy(), D, levels=D.max() * np.array(level_fractions),
               colors=[color], linewidths=1.2)


def main(idxs, split="val", same_plot=True):
    _, dec = load_vae(vae, hidden_dims, latent_dim, act, input_dim)
    X = load_dataset("mnist_bin")[split][idxs]
    post, means, g = posteriors(X, dec)
    colors = plt.cm.tab20(np.linspace(0, 1, len(idxs)))

    if same_plot:
        # all posteriors on one plane, each labelled with its index at its mean
        fig, ax = plt.subplots(figsize=(8, 8))
        for k, idx in enumerate(idxs):
            draw(ax, g, post[k], colors[k])
            ax.text(means[k, 0].item(), means[k, 1].item(), str(idx), fontsize=8,
                    ha="center", va="center", color=colors[k], fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    else:
        # one panel per image, all on the same axes, with the posterior mean in the title
        cols = min(6, len(idxs))
        rows = math.ceil(len(idxs) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False,
                                 sharex=True, sharey=True)
        for k, idx in enumerate(idxs):
            ax = axes[k // cols, k % cols]
            draw(ax, g, post[k], "tab:green")
            ax.set_title(f"{idx}: z = ({means[k, 0]:.2f}, {means[k, 1]:.2f})", fontsize=8)
            ax.set_aspect("equal")
        for ax in axes.flat[len(idxs):]:
            ax.axis("off")

    for idx, m in zip(idxs, means.tolist()):
        print(idx, [round(v, 2) for v in m])
    plt.savefig(f"posterior_map_{split}.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main(idxs=list(range(30)), split="val", same_plot=True)