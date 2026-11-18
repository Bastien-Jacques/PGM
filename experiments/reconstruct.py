"""Original images versus reconstructions of a trained 2D-latent VAE, with the amortized FFG or the optimal flow."""
import torch
import matplotlib.pyplot as plt
from utils import load_vae, load_dataset, flow_name, load_flow

# settings
vae_name = "vae_ffg_h100_tanh_seed0"
input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, act = 784, [100], [100], 2, "tanh"
split = "val"                 # "train" or "val"
idxs = range(0,30)           # images shown, one column each
seed = 0                      # for the sampled latent codes and pixels


def plot_rows(rows, fname=None):
    """One row per (label, images) pair, one column per image."""
    fig, axes = plt.subplots(len(rows), len(idxs), figsize=(1.2 * len(idxs), 1.3 * len(rows)), squeeze=False)
    for r, (name, imgs) in enumerate(rows):
        for c in range(len(idxs)):
            axes[r, c].imshow(imgs[c].view(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=8, family="serif")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if fname is not None:
        plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.show()


def main(family):
    torch.manual_seed(seed)
    enc, dec = load_vae(vae_name, hidden_dim_enc=hidden_dim_enc, hidden_dim_dec=hidden_dim_dec, latent_dim=latent_dim, act=act, input_dim=input_dim)
    X = load_dataset("mnist_bin")[split][idxs]        # (len(idxs), 784), in the order of idxs

    with torch.no_grad():
        if family == "ffg":
            mu, logsig = enc(X)
            z = mu + torch.exp(logsig) * torch.randn_like(mu)                 # one code per image from q(z|x)
            rows = [("original", X), ("grey, z = mu", torch.sigmoid(dec(mu)))]
        elif family == "flow":
            flows = {idx: load_flow(flow_name(idx, vae_name, split)) for idx in idxs}
            z = torch.cat([flows[idx].sample(1)[0] for idx in idxs])           # one code per image, each from its own flow
            rows = [("original", X)]

        mean_at_z = torch.sigmoid(dec(z))
        rows += [(f"grey, z ~ q_{family}", mean_at_z),
                 ("sample", torch.bernoulli(mean_at_z))]                       # binary image drawn from p(x|z)

    plot_rows(rows, f"recon_{family}_{vae_name}_{split}.png")


if __name__ == "__main__":
    main("ffg")

# How to read the plot:
# first row: the original binarized images
# "grey, z = mu" (ffg only): decoded mean sigma(dec(mu)) at the centre of the encoder's Gaussian
# "grey, z ~ q": one latent code drawn from q (encoder Gaussian or optimal flow), decoded mean sigma(dec(z))
# "sample": binary image drawn from p(x|z), each pixel black with probability sigma(dec(z))