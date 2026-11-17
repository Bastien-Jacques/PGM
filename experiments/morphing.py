"""Morphing between two images: x_alpha = (1 - alpha) x_0 + alpha x_N, and the exact posteriors along the path."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from grid import make_grid, posterior_on_grid
from utils import load_vae, load_dataset

# settings
vae = "vae_ffg_h100_tanh_seed0"
input_dim, hidden_dims, latent_dim, act = 784, [100], 2, "tanh"
split = "val"
i0, iN = 0, 1                      # endpoint images   5-22
N = 1000                            # number of steps: the path has N + 1 images
L, n = 6.0, 300                     # global grid, large enough for the image far from the origin
coarse_step = 2                     # keep one image every coarse_step for the quick check
alphas_shown = [0.0, 0.25, 0.5, 0.75, 1.0]
level_fractions = [0.1, 0.5]


def make_path(x0, xN, N):
    """Linear interpolation in pixel space; returns alpha (N + 1,) and the images (N + 1, 784)."""
    alpha = torch.linspace(0, 1, N + 1)
    X_path = (1 - alpha[:, None]) * x0 + alpha[:, None] * xN
    return alpha, X_path


@torch.no_grad()
def second_peak_ratio(post, kernel=7):
    """Height of the second local maximum of each posterior divided by the height of the first."""
    post = post.view(-1, 1, n, n)
    # a grid point is a local maximum if it equals the maximum over its neighbourhood
    is_peak = post == F.max_pool2d(post, kernel, stride=1, padding=kernel // 2)
    heights = torch.where(is_peak, post, torch.zeros_like(post)).flatten(1)
    top2 = heights.topk(2, dim=1).values
    return top2[:, 1] / top2[:, 0]


def draw(ax, g, dens, color):
    """Contour lines of one posterior, in a single colour."""
    D = dens.view(n, n).T.numpy()                                # contour expects (y, x) indexing
    ax.contour(g.numpy(), g.numpy(), D, levels=D.max() * np.array(level_fractions),
               colors=[color], linewidths=1.2)


def main():
    _, dec = load_vae(vae, hidden_dims, latent_dim, act, input_dim)
    X = load_dataset("mnist_bin")[split]
    alpha, X_path = make_path(X[i0], X[iN], N)

    # sanity check: the path starts and ends exactly at the two images
    assert torch.equal(X_path[0], X[i0]) and torch.equal(X_path[-1], X[iN])

    # a few frames of the morphing
    frames = X_path[::N //10]
    fig, axes = plt.subplots(1, len(frames), figsize=(1.2 * len(frames), 1.4))
    for ax, img, a in zip(axes, frames, alpha[::N // 10]):
        ax.imshow(img.view(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{a:.1f}", fontsize=8)
        ax.axis("off")
    plt.savefig(f"morph_frames_{i0}_{iN}.png", dpi=200, bbox_inches="tight")
    plt.show()

    # exact posteriors on a coarse version of the path
    with torch.no_grad():
        Z, lp, lda = make_grid(L, n)
        a_coarse = alpha[::coarse_step]
        post = posterior_on_grid(X_path[::coarse_step], Z, dec, lp, lda)   # (len(a_coarse), n * n)
    ratio = second_peak_ratio(post)
    g = torch.linspace(-L, L, n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # bimodality along the path: close to zero for a single bump, close to one for two equal bumps
    ax1.plot(a_coarse.numpy(), ratio.numpy(), "o-")
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel("second peak / first peak")
    ax1.set_ylim(0, 1.05)

    # posteriors for a few values of alpha, coloured along the path
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas_shown)))
    for a, c in zip(alphas_shown, colors):
        k = int(round(a * (len(a_coarse) - 1)))                 # closest point of the coarse path
        draw(ax2, g, post[k], c)
        ax2.plot([], [], color=c, label=rf"$\alpha$ = {a_coarse[k]:.2f}")
    ax2.set_aspect("equal"); ax2.legend(fontsize=8)
    ax2.set_xlabel("$z_1$"); ax2.set_ylabel("$z_2$")

    plt.savefig(f"morph_check_{i0}_{iN}.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()