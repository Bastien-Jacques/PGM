"""Morphing between two images, x_alpha = (1 - alpha) x_0 + alpha x_N: frames of the path and bimodality of the exact posterior."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from grid import make_grid, posterior_on_grid
from utils import load_vae, load_dataset
from grid import make_box_grid
import os


# settings
vae = "vae_ffg_h100_tanh_seed0"
input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, act = 784, [100], [100], 2, "tanh"
split = "val"
i0, iN = 0, 1                       # endpoint images
N = 1000                            # number of steps: the path has N + 1 images
L, n = 6.0, 300                     # global grid, large enough for every posterior along the path
step = 2                            # posteriors are computed on one image every step
n_frames = 11                       # images shown in the strip
chunk = 100                         # images per batch when computing posteriors, to bound memory

# paper style
mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 300,
})


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


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", bbox_inches="tight")


def main():
    _, dec = load_vae(vae, hidden_dim_enc=hidden_dim_enc, hidden_dim_dec=hidden_dim_dec,
                      latent_dim=latent_dim, act=act, input_dim=input_dim)
    X = load_dataset("mnist_bin")[split]
    alpha, X_path = make_path(X[i0], X[iN], N)

    # sanity check: the path starts and ends exactly at the two images
    assert torch.equal(X_path[0], X[i0]) and torch.equal(X_path[-1], X[iN])

    # strip of frames along the path, dark digits on a white background
    idx = torch.linspace(0, N, n_frames).round().long()
    fig, axes = plt.subplots(1, n_frames, figsize=(7.0, 0.95))
    for ax, k in zip(axes, idx):
        ax.imshow(X_path[k].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.6")
        ax.set_xlabel(rf"${alpha[k]:.1f}$", fontsize=8, labelpad=2)
    axes[0].set_ylabel(r"$\alpha$", rotation=0, fontsize=9, labelpad=8, va="center")
    fig.subplots_adjust(wspace=0.08)
    save(fig, f"morph_frames_{i0}_{iN}")

    # bimodality of the exact posterior along the path, computed by chunks to bound memory
    with torch.no_grad():
        Z, lp, lda = make_grid(L, n)
        a_sub, X_sub = alpha[::step], X_path[::step]
        ratio = torch.cat([second_peak_ratio(posterior_on_grid(X_sub[i:i + chunk], Z, dec, lp, lda))
                           for i in range(0, len(X_sub), chunk)])
    alpha_c = a_sub[ratio.argmax()].item()             # two bumps of equal height: the mode switches region here
    print(f"largest bimodality at alpha = {alpha_c:.3f}")

    alphas = [0.0, 0.1, 0.2, 0.3, round(alpha_c, 2), 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    window = (-1.6, 1.6, -0.5, 1.6)                    # (z1 min, z1 max, z2 min, z2 max), covers both regions
    frames_and_posteriors(dec, X_path, alpha, alphas, window)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(a_sub.numpy(), ratio.numpy(), color="tab:green", lw=1.4)
    ax.axvline(alpha_c, ls=":", color="0.4", lw=0.9)
    ax.text(alpha_c, 1.02, r"$\alpha_c$", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("second / first peak height")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, f"morph_bimodality_{i0}_{iN}")
    plt.show()

    save_posteriors_svg(dec, X_path, alpha, alphas, window)


def frames_and_posteriors(dec, X_path, alpha, alphas, window, n_box=300):
    """Images of the path (top) and their exact posteriors on a common latent window (bottom)."""
    idx = [int(round(a * (len(alpha) - 1))) for a in alphas]
    a1, b1, a2, b2 = window
    with torch.no_grad():
        Z, lp, lda, g1, g2 = make_box_grid(a1, b1, a2, b2, n_box)
        post = posterior_on_grid(X_path[idx], Z, dec, lp, lda)                # (len(alphas), n_box ** 2)

    fig, axes = plt.subplots(2, len(idx), figsize=(7.0, 1.75),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08, "wspace": 0.08})
    for col, (k, p) in enumerate(zip(idx, post)):
        ax_img, ax_post = axes[0, col], axes[1, col]
        ax_img.imshow(X_path[k].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax_img.set_title(rf"${alpha[k]:.2f}$", fontsize=8, pad=2)

        # each posterior with its own levels: they are very peaked, a common scale would hide most of them
        D = p.view(n_box, n_box).T.numpy()                                      # contourf expects (y, x) indexing
        ax_post.contourf(g1.numpy(), g2.numpy(), D, levels=D.max() * np.array([0.05, 0.25, 0.5, 0.75, 1.0]),
                         cmap="Greens")
        ax_post.set_xlim(a1, b1); ax_post.set_ylim(a2, b2); ax_post.set_aspect("equal")

        for ax in (ax_img, ax_post):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.6")
    axes[0, 0].set_ylabel(r"$x_\alpha$", rotation=0, fontsize=9, labelpad=10, va="center")
    axes[1, 0].set_ylabel(r"$p_\theta(z \mid x_\alpha)$", fontsize=8, labelpad=4)
    save(fig, f"morph_posteriors_{i0}_{iN}")


def save_posteriors_svg(dec, X_path, alpha, alphas, window, n_box=300, folder="posteriors_svg"):
    """One SVG file per posterior, all on the same latent window, without axes, for assembling a poster."""
    os.makedirs(folder, exist_ok=True)
    idx = [int(round(a * (len(alpha) - 1))) for a in alphas]
    a1, b1, a2, b2 = window
    with torch.no_grad():
        Z, lp, lda, g1, g2 = make_box_grid(a1, b1, a2, b2, n_box)
        post = posterior_on_grid(X_path[idx], Z, dec, lp, lda)

    for k, p in zip(idx, post):
        fig, ax = plt.subplots(figsize=(1.2, 1.2 * (b2 - a2) / (b1 - a1)))       # same aspect ratio as the window
        D = p.view(n_box, n_box).T.numpy()                                        # contourf expects (y, x) indexing
        ax.contourf(g1.numpy(), g2.numpy(), D, levels=D.max() * np.array([0.05, 0.25, 0.5, 0.75, 1.0]),
                    cmap="Greens")
        ax.set_xlim(a1, b1); ax.set_ylim(a2, b2); ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(os.path.join(folder, f"posterior_alpha{alpha[k]:.3f}.svg"),
                    bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)

if __name__ == "__main__":
    main()