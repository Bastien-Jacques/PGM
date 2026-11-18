"""Morphing analysis: the map alpha -> nu*(x_alpha) of the optimal FFG versus the encoder, in the latent plane and through W2."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from grid import make_grid, posterior_on_grid
from utils import load_artifact, load_dataset, load_vae
from objectives import elbo_ffg_batch

# settings
vae = "vae_ffg_h100_tanh_seed0"
input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, act = 784, [100], [100], 2, "tanh"
data_name = "morph_val0_1_N1000"
ffg_runs = [f"optffg_{data_name}_initprior_K20_seed0:v1"]      # best of K random random restarts from the prior for every omage of the morphing
                                                       # every image started from the optimum of x_N
L, n = 6.0, 300                                        # grid for the endpoint posteriors
ellipse_every = 50                                     # draw one optimal Gaussian every ellipse_every steps
jump_factor = 10.0                                     # a step is a jump if its W2 exceeds jump_factor times the median


def load_run(spec):
    """Load a run saved as an artifact; spec is "name" (latest version) or "name:vK"."""
    name, _, version = spec.partition(":")
    return torch.load(os.path.join(load_artifact(name, version or "latest"), f"{name}.pt"))

def w2_neighbours(mu, logsig):
    """W2 between consecutive diagonal Gaussians: sqrt(|mu - mu'|^2 + |sigma - sigma'|^2)."""
    sig = torch.exp(logsig)
    return torch.sqrt(((mu[1:] - mu[:-1]) ** 2).sum(-1) + ((sig[1:] - sig[:-1]) ** 2).sum(-1))


def draw_posterior(ax, g, dens):
    D = dens.view(n, n).T.numpy()                      # contour expects (y, x) indexing
    ax.contour(g.numpy(), g.numpy(), D, levels=D.max() * np.array([0.05, 0.2, 0.5, 0.8]),
               cmap="Greens", linewidths=1.2)


def main():
    path = load_dataset(data_name)
    alpha, X = path["alpha"], path["X"]
    enc, dec = load_vae(vae, hidden_dim_enc=hidden_dim_enc, hidden_dim_dec=hidden_dim_dec, latent_dim=latent_dim, act=act, input_dim=input_dim)

    # optimal FFG: for each alpha, keep the branch with the larger ELBO
    runs = [load_run(r) for r in ffg_runs]
    elbos = torch.stack([r["elbo"] for r in runs])                     # (branches, N + 1)
    best = elbos.argmax(0)                                             # (N + 1,)
    pick = lambda key: torch.stack([r[key] for r in runs])[best, torch.arange(len(alpha))]
    mu_s, logsig_s = pick("mu"), pick("logsig")                        # (N + 1, 2) each

    # amortized FFG on the same path
    with torch.no_grad():
        mu_a, logsig_a = enc(X)
        elbo_a = torch.stack([elbo_ffg_batch(X, dec, mu_a, logsig_a, S=100) for _ in range(20)]).mean(0)   # (N + 1,)  ELBO of the encoder's Gaussian on every image, 
                                                                                                           # averaged over independent estimates
    elbo_s = elbos.max(0).values                                       # ELBO of the selected optimal FFG
    gap = elbo_s - elbo_a                                              # amortization gap along the path
    w2_s = w2_neighbours(mu_s, logsig_s)
    w2_a = w2_neighbours(mu_a, logsig_a)
    a_mid = 0.5 * (alpha[1:] + alpha[:-1])                             # W2 lives between two consecutive images
    alpha_c = a_mid[w2_s.argmax()].item()                              # position of the largest jump of nu*
    print(f"jump at alpha = {alpha_c:.3f}, max amortization gap = {gap.max():.2f} at alpha = {alpha[gap.argmax()]:.3f}")
    jumps = w2_s > jump_factor * w2_s.median()                         # every step large enough to cut the path
    print("jumps of nu* at alpha =", [round(a, 3) for a in a_mid[jumps].tolist()])

    # endpoint posteriors
    with torch.no_grad():
        Z, lp, lda = make_grid(L, n)
        post = posterior_on_grid(X[[0, -1]], Z, dec, lp, lda)
    g = torch.linspace(-L, L, n)

    # latent plane
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_posterior(ax, g, post[0]); draw_posterior(ax, g, post[1])
    cmap = plt.cm.viridis
    for i in range(0, len(alpha), ellipse_every):                       # optimal Gaussians, coloured by alpha
        s = torch.exp(logsig_s[i])
        ax.add_patch(Ellipse(mu_s[i].tolist(), 2 * s[0].item(), 2 * s[1].item(),
                             facecolor=cmap(alpha[i].item()), alpha=0.25, edgecolor="none"))
    segments = np.split(np.arange(len(alpha)), torch.nonzero(jumps).flatten().add(1).tolist())
    for seg in segments:                                                # centre path, cut at the jumps
        ax.plot(mu_s[seg, 0], mu_s[seg, 1], color="tab:blue", lw=1.5)
    ax.plot(mu_a[:, 0], mu_a[:, 1], color="tab:red", lw=1.5, ls="--", label="encoder")
    ax.plot([], [], color="tab:blue", label="optimal FFG")
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, label=r"$\alpha$")
    ax.set_aspect("equal"); ax.legend(fontsize=9)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    plt.savefig(f"morph_latent_{data_name}.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ELBO of the two branches and W2 between neighbours
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    
    ax1.plot(alpha, gap, color="tab:purple")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_ylabel(r"amortization gap $\mathcal{L}(q^\star) - \mathcal{L}(q_{\mathrm{amort}})$")
    for ax in (ax1, ax2):
        ax.axvline(alpha_c, ls=":", color="gray")        # jump of the optimal FFG
    tau = 5e-3                                           # floor of the plot: variations of magnitude under tau are flattened
    ax2.semilogy(a_mid, w2_s.clamp(min=tau), label=r"optimal FFG $\nu^\star$")
    ax2.semilogy(a_mid, w2_a.clamp(min=tau), label="encoder", ls="--")
    ax2.set_ylim(bottom=0.8 * tau)
    ax2.set_xlabel(r"$\alpha$"); ax2.set_ylabel(r"$W_2$ between neighbours"); ax2.legend(fontsize=9)
    plt.savefig(f"morph_w2_{data_name}.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()