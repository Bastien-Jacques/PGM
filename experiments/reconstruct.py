"""Original images versus reconstructions of a trained 2D-latent VAE."""
import os
import torch
import matplotlib.pyplot as plt
from model import load_vae

# settings
run_name = "vae_ffg_h200-200_tanh_seed0"
artifact_dir = os.path.join("artifacts", f"{run_name}-v0")
input_dim, hidden_dims, latent_dim, act = 784, [200,200], 2, "tanh"


split = "val"                 # "train" or "val"
idxs = list(range(10))        # images shown
seed = 0                      # for the sampled latent codes and pixels


def main():
    torch.manual_seed(seed)
    enc, dec = load_vae(artifact_dir, run_name, hidden_dims, latent_dim, act, input_dim)
    X = torch.load("mnist_bin.pt")[split][idxs]            # (n, 784), binary images

    with torch.no_grad():
        mu, logsig = enc(X)
        mean_at_mu = torch.sigmoid(dec(mu))                # decoded mean at the centre of q(z|x)
        z = mu + torch.exp(logsig) * torch.randn_like(mu)  # one latent code drawn from q(z|x)
        mean_at_z = torch.sigmoid(dec(z))
        sample = torch.bernoulli(mean_at_z)                # binary image drawn from p(x|z)

    rows = [("original", X), ("grey, z = mu", mean_at_mu),
            ("grey, z ~ q", mean_at_z), ("sample", sample)]
    fig, axes = plt.subplots(len(rows), len(idxs), figsize=(1.2 * len(idxs), 1.3 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        for c in range(len(idxs)):
            ax = axes[r, c]
            ax.imshow(imgs[c].view(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=8, family="serif")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f"recon_{run_name}_{split}.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

## How to read the plot : ##

#the first row just plot the original data (binarized MNIST)
#the second uses the latent vector z = mu where mu is the parameter of the mean produced by the encoder then produces the picture in grey-levels using x^ = sigma(dec(z))
#the third row is the plot picking a sample z ~ N(mu, sigma); where mu, sigma = enc(x); in the latent space and then plots the reconstructed image in grey levels using x^ = sigma(dec(z))
#the last is obtained by picking a z ~ N(mu, sigma) anf then plots in binarized, using pixel = black if sigma(dec(z)) >= 0,5; withe else