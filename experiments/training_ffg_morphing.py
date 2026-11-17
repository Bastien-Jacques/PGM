"""Optimal FFG on every image of a dataset artifact (e.g. a morphing path), all images optimized in one run."""
import argparse
import os

import torch
import wandb

from objectives import elbo_ffg_batch
from utils import load_artifact, load_dataset, load_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default="morph_val0_1_N1000")   # dataset artifact holding {"X", "alpha", ...}
p.add_argument("--init", type=str, default="prior")                # "prior", or the name of an optimal FFG run to start from
p.add_argument("--hidden_dims", type=int, nargs="+", default=[100])
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--sample_size", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

torch.manual_seed(args.seed)

name_vae = "vae_ffg_h100_tanh_seed0"
init_tag = "prior" if args.init == "prior" else args.init.split("_")[-2]     # e.g. "val0" for optffg_..._val0_seed0
run_name = f"optffg_{args.data}_init{init_tag}_seed{args.seed}"
config = vars(args) | {"model": "optimal_ffg", "vae": name_vae}
run = wandb.init(project=args.project, name=run_name, config=config)

# images of the path and frozen decoder
data = load_dataset(args.data)
X = data["X"].to(device)                                               # (B, 784)
B = X.shape[0]
_, dec = load_vae(name_vae, args.hidden_dims, args.latent_dim, "tanh", 784)
dec = dec.to(device).requires_grad_(False)

# free variational parameters, one row per image, all starting from the same point
if args.init == "prior":
    mu0, logsig0 = torch.zeros(args.latent_dim), torch.zeros(args.latent_dim)
else:
    saved = torch.load(os.path.join(load_artifact(args.init, "latest"), f"{args.init}.pt"))
    mu0, logsig0 = saved["mu"].reshape(-1), saved["logsig"].reshape(-1)
mu = mu0.to(device).expand(B, -1).clone().requires_grad_(True)       # (B, 2)
logsig = logsig0.to(device).expand(B, -1).clone().requires_grad_(True)
optimizer = torch.optim.Adam([mu, logsig], lr=args.lr)

for step in range(args.steps):
    elbo = elbo_ffg_batch(X, dec, mu, logsig, args.sample_size)       # (B,)
    loss = -elbo.sum()                                                # images are independent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        wandb.log({"neg_elbo_per_image": -elbo.mean().item()}, step=step)

# final ELBO of each image, averaged over independent estimates to keep memory bounded
with torch.no_grad():
    final_elbo = torch.stack([elbo_ffg_batch(X, dec, mu, logsig, S=100) for _ in range(50)]).mean(0)   # (B,)

path = f"/kaggle/working/{run_name}.pt"
torch.save({"alpha": data["alpha"], "mu": mu.detach().cpu(), "logsig": logsig.detach().cpu(),
            "elbo": final_elbo.cpu()}, path)
art = wandb.Artifact(name=run_name, type="ffg", metadata=config)
art.add_file(path)
run.log_artifact(art)
run.finish()