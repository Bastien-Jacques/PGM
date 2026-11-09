"""Figure 2 of Cremer et al. (2018): exact posterior and approximations for a 2D-latent VAE."""
import math
import os
from utils import load_artifact, load_vae, load_flow, load_dataset

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grid import make_grid, make_box_grid, posterior_on_grid



global_L, global_N = 4.0, 200       # global grid, used only to locate the posterior
local_N = 200                       # local grid, used for the plot and the KL
pad = 0.6                           # margin around the distributions, as a fraction of the window
level_fractions = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
eps = 1e-30                         # avoids log(0) in the KL



def window_from_posterior(post, Z, log_dA, k=4.0):
    #Window mean +/- k standard deviations of the posterior, computed on the grid
    w = post * math.exp(log_dA)
    w = w / w.sum()
    m = (w[:, None] * Z).sum(0)
    sd = torch.sqrt((w[:, None] * (Z - m) ** 2).sum(0))
    return ((m[0] - k * sd[0]).item(), (m[0] + k * sd[0]).item(),
            (m[1] - k * sd[1]).item(), (m[1] + k * sd[1]).item())


def window_from_gaussian(mu, logsig, k=4.0):
    #Window mu +/- k sigma of a diagonal Gaussian
    s = torch.exp(logsig)
    return ((mu[0] - k * s[0]).item(), (mu[0] + k * s[0]).item(),
            (mu[1] - k * s[1]).item(), (mu[1] + k * s[1]).item())


def union_window(w1, w2):
    #Smallest window containing both windows (a1, b1, a2, b2)
    return (min(w1[0], w2[0]), max(w1[1], w2[1]), min(w1[2], w2[2]), max(w1[3], w2[3]))


def pad_window(w, factor=pad):
    #Square window around w, enlarged by a fraction of its size
    a1, b1, a2, b2 = w
    c1, c2 = (a1 + b1) / 2, (a2 + b2) / 2
    half = max(b1 - a1, b2 - a2) / 2 * (1 + factor)
    return c1 - half, c1 + half, c2 - half, c2 + half


def gauss_density(Z, mu, logsig):
    #Density of N(mu, diag(sigma^2)) at the points Z (K, 2)
    u = (Z - mu) / torch.exp(logsig)
    log_q = -0.5 * (u ** 2).sum(-1) - logsig.sum() - math.log(2 * math.pi)
    return torch.exp(log_q)


def kl_on_grid(q, post, log_dA):
    #KL(q || p(.|x)) by quadrature on the grid; equals log p(x) - ELBO(x) for this image
    return ((q * (torch.log(q + eps) - torch.log(post + eps))).sum() * math.exp(log_dA)).item()


def draw_density(ax, g1, g2, dens, cmap):
    #Contour lines of a density, coloured by level as in Cremer et al
    n = len(g1)
    D = dens.view(n, n).T.cpu().numpy()        # contour expects (y, x) indexing
    ax.contour(g1.cpu().numpy(), g2.cpu().numpy(), D,
               levels=D.max() * level_fractions, cmap=cmap, linewidths=1.6)


def plot_panel(ax, g1, g2, post, q=None):
    #Exact posterior in green, approximation in blue on top, no frame
    draw_density(ax, g1, g2, post, "Greens")
    if q is not None:
        draw_density(ax, g1, g2, q, "Blues")
    ax.set_aspect("equal")
    ax.axis("off")



# settings #

input_dim, hidden_dims, latent_dim, act = 784, [100], 2, "tanh"
split = "val"      

vae = "vae_ffg_h100_tanh_seed0"
enc, dec = load_vae(vae, hidden_dims, latent_dim, act, input_dim)
data = load_dataset("mnist_bin")
X = data[split]

@torch.no_grad()
def second_peak_ratio(X, dec, L=4.0, n=200, kernel=7, chunk=200):
    #Height of the second local maximum of the posterior divided by the height of the first
    Z, lp, lda = make_grid(L, n)
    ratios = []
    for i in range(0, X.shape[0], chunk):
        post = posterior_on_grid(X[i:i + chunk], Z, dec, lp, lda).view(-1, 1, n, n)
        # a grid point is a local maximum if it equals the maximum over its neighbourhood
        is_peak = post == F.max_pool2d(post, kernel, stride=1, padding=kernel // 2)
        heights = torch.where(is_peak, post, torch.zeros_like(post)).flatten(1)
        top2 = heights.topk(2, dim=1).values       # the two highest local maxima
        ratios.append(top2[:, 1] / top2[:, 0])
    return torch.cat(ratios)

r = second_peak_ratio(X[:2000], dec)
order = r.argsort(descending=True)
#print([(i.item(), round(r[i].item(), 2)) for i in order[:20]])   # bimodal candidates idx

def flow_name(idx, n_layers=8, hidden=32, seed=0):
    return f"optflow_L{n_layers}_h{hidden}_{vae}_{split}{idx}_seed{seed}"


def main(idxs: list):

    row_names = ["True posterior", "Amortized FFG", "Optimal Flow"]
    fig, axes = plt.subplots(len(row_names), len(idxs), figsize=(3 * len(idxs), 3 * len(row_names)))

    with torch.no_grad():
        Z, lp, lda = make_grid(global_L, global_N)
        for col, idx in enumerate(idxs):
            x = X[idx]
            mu, logsig = (t[0] for t in enc(x[None]))

            # one window per column, covering the posterior and the encoder's Gaussian
            post_global = posterior_on_grid(x[None], Z, dec, lp, lda)[0]
            window = union_window(window_from_posterior(post_global, Z, lda),
                                  window_from_gaussian(mu, logsig))
            a1, b1, a2, b2 = pad_window(window)

            # exact posterior and q on the local grid
            Zl, lpl, ldal, g1, g2 = make_box_grid(a1, b1, a2, b2, local_N)
            post = posterior_on_grid(x[None], Zl, dec, lpl, ldal)[0]
            q_amort = gauss_density(Zl, mu, logsig)
            q_flow = torch.exp(load_flow(flow_name(idx)).log_prob(Zl))

            plot_panel(axes[0, col], g1, g2, post)
            plot_panel(axes[1, col], g1, g2, post, q_amort)
            plot_panel(axes[2, col], g1, g2, post, q_flow)
            axes[0, col].set_title(f"{idx}", fontsize=9)
            axes[1, col].set_title(f"KL = {kl_on_grid(q_amort, post, ldal):.2f}", fontsize=9)
            axes[2, col].set_title(f"KL = {kl_on_grid(q_flow, post, ldal):.2f}", fontsize=9)

    for r, name in enumerate(row_names):
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, family="serif")
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.savefig(f"fig2_{vae}_{split}.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main(idxs = [0, 1, 2, 1056, 394, 221, 291 ])