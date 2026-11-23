"""Figure 2 of Cremer et al. (2018): exact posterior and approximations for a 2D-latent VAE
import math
import os
from utils import load_vae, load_flow, load_dataset, flow_name, load_optimal_ffg

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

input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, act = 784, [100], [100], 2, "tanh"
split = "val"      

vae = "vae_ffg_h100_tanh_seed0"
enc, dec = load_vae(vae, hidden_dim_enc=hidden_dim_enc, hidden_dim_dec=hidden_dim_enc, latent_dim=latent_dim, act=act, input_dim=input_dim)
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



def main(idxs: list):

    row_names = ["True posterior", "Amortized FFG", "Optimal FFG", "Optimal Flow"]
    fig, axes = plt.subplots(len(row_names), len(idxs), figsize=(3 * len(idxs), 3 * len(row_names)))

    with torch.no_grad():
        Z, lp, lda = make_grid(global_L, global_N)
        for col, idx in enumerate(idxs):
            x = X[idx]
            mu, logsig = (t[0] for t in enc(x[None]))
            mu_opt, logsig_opt = load_optimal_ffg(idx, vae, split)

            # one window per column, covering the posterior and both Gaussians
            post_global = posterior_on_grid(x[None], Z, dec, lp, lda)[0]
            window = union_window(window_from_posterior(post_global, Z, lda),
                                  window_from_gaussian(mu, logsig))
            window = union_window(window, window_from_gaussian(mu_opt, logsig_opt))
            a1, b1, a2, b2 = pad_window(window)

            # exact posterior and the three approximations on the local grid
            Zl, lpl, ldal, g1, g2 = make_box_grid(a1, b1, a2, b2, local_N)
            post = posterior_on_grid(x[None], Zl, dec, lpl, ldal)[0]
            q_amort = gauss_density(Zl, mu, logsig)
            q_ffg = gauss_density(Zl, mu_opt, logsig_opt)
            q_flow = torch.exp(load_flow(flow_name(idx, vae, split, annealing=True)).log_prob(Zl))

            plot_panel(axes[0, col], g1, g2, post)
            axes[0, col].set_title(f"{idx}", fontsize=9)
            for r, q in enumerate([q_amort, q_ffg, q_flow], start=1):
                plot_panel(axes[r, col], g1, g2, post, q)
                axes[r, col].set_title(f"KL = {kl_on_grid(q, post, ldal):.2f}", fontsize=9)

    for r, name in enumerate(row_names):
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, family="serif")
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.savefig(f"fig2_{vae}_{split}.pdf", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main(idxs = [0, 1, 2, 1056, 394, 221, 291 ])
"""

"""True posterior vs shared-flow q along the morphing path, around the mode switch"""
import math as m

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import Flow2D
from utils import load_dataset, load_run, load_vae

data_name = "morph_val0_1_N1000"
flow_run = f"optsharedflow_{data_name}_L8_h32_K10_seed0"
cont_run = f"cont_{flow_run}_steps300"
alphas = [0.20, 0.28, 0.31, 0.33, 0.35, 0.40, 0.50]
G = 800                                   # fine grid resolution
margin = 0.5                              # margin around the posterior mass
levels_rel = [0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]    # contour levels relative to the max density


@torch.no_grad()
def log_joint_grid(X, dec, zz, chunk=20000):
    """log p(x, z) for each image of X on each grid point, computed by chunks of the grid."""
    out = []
    for zc in zz.split(chunk):
        logits = dec(zc)
        ll = X @ F.logsigmoid(logits).T + (1 - X) @ F.logsigmoid(-logits).T
        out.append(ll - 0.5 * (zc ** 2).sum(-1) - m.log(2 * m.pi))
    return torch.cat(out, 1)                                         # (n_images, n_grid)


def grid(lo, hi, G):
    g1, g2 = torch.linspace(lo[0], hi[0], G), torch.linspace(lo[1], hi[1], G)
    Z1, Z2 = torch.meshgrid(g1, g2, indexing="xy")
    cell = (g1[1] - g1[0]) * (g2[1] - g2[0])
    return Z1, Z2, torch.stack([Z1.flatten(), Z2.flatten()], 1), cell.item()


def draw(ax, Z1, Z2, dens, cmap):
    d = dens.numpy()
    ax.contour(Z1.numpy(), Z2.numpy(), d, levels=d.max() * np.array(levels_rel), cmap=cmap, linewidths=1.0)

@torch.no_grad()
def kl_along_path(data, dec, c, ffg, lo=(-4, -4), hi=(4, 4), G=400):
    """KL(q || posterior) = log p(x) - ELBO for every image of the path."""
    _, _, zz, cell = grid(list(lo), list(hi), G)
    log_px = torch.logsumexp(log_joint_grid(data["X"], dec, zz), 1) + m.log(cell)
    kl_flow = log_px - torch.maximum(c["elbo_fwd"], c["elbo_bwd"])
    kl_ffg = log_px - ffg["elbo"]
    return kl_flow, kl_ffg

@torch.no_grad()
def main():
    data, run, c = load_dataset(data_name), load_run(flow_run), load_run(cont_run)
    _, dec = load_vae("vae_ffg_h100_tanh_seed0", hidden_dim_enc=[100], hidden_dim_dec=[100],
                      latent_dim=2, act="tanh", input_dim=784)
    T = Flow2D(n_layers=run["n_layers"], hidden=run["hidden"])
    T.load_state_dict(run["flow_state"])

    idx = [(data["alpha"] - a).abs().argmin().item() for a in alphas]
    X = data["X"][idx]

    # common window around the mass of the true posteriors
    _, _, zz, _ = grid([-4, -4], [4, 4], 200)
    post = torch.softmax(log_joint_grid(X, dec, zz), 1)
    keep = zz[(post > 1e-4 * post.max(1, keepdim=True).values).any(0)]
    lo, hi = (keep.min(0).values - margin).tolist(), (keep.max(0).values + margin).tolist()

    Z1, Z2, zz, cell = grid(lo, hi, G)
    log_joint = log_joint_grid(X, dec, zz)
    log_px = torch.logsumexp(log_joint, 1) + m.log(cell)
    post = torch.exp(log_joint - log_px[:, None])                    # normalized true posterior densities

    rows = [("True posterior", None, None), ("Shared flow, forward", c["nu_fwd"], c["elbo_fwd"]),
            ("Shared flow, backward", c["nu_bwd"], c["elbo_bwd"])]
    fig, axes = plt.subplots(len(rows), len(alphas), figsize=(2.0 * len(alphas), 2.2 * len(rows)))
    for j, n in enumerate(idx):
        p_true = post[j].view(G, G)
        for i, (name, nu_all, elbo_all) in enumerate(rows):
            ax = axes[i, j]
            draw(ax, Z1, Z2, p_true, "Greens")
            if nu_all is None:
                ax.set_title(rf"$\alpha = {data['alpha'][n]:.2f}$", fontsize=9)
            else:
                T.mu.data, T.logsigma.data = nu_all[n][:2].clone(), nu_all[n][2:].clone()
                draw(ax, Z1, Z2, torch.exp(T.log_prob(zz)).view(G, G), "Blues")
                ax.set_title(f"KL = {log_px[j].item() - elbo_all[n].item():.2f}", fontsize=9)
            ax.set_aspect("equal")
            ax.axis("off")
            if j == 0:
                ax.text(-0.15, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=10, family="serif")
    fig.tight_layout()
    fig.savefig("sharedflow_vs_true.pdf", bbox_inches="tight")
  

    ffg = load_run(f"optffg_{data_name}_initprior_K20_seed0:v1")
    kl_flow, kl_ffg = kl_along_path(data, dec, c, ffg)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(data["alpha"], kl_ffg, color="#a50f15", lw=1.0, label="optimal FFG")
    ax.plot(data["alpha"], kl_flow, color="#2171b5", lw=1.0, label="shared flow (best of the two sweeps)")
    ax.axhline(m.log(2), color="gray", lw=0.6, ls="--")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"KL$(q \,\|\, p_\theta(z \mid x))$ (nats)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("kl_path_ffg_vs_sharedflow.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()