#Optimal FFG: per-image F-VI with a factorized Gaussian, decoder frozen, all images optimized together
import argparse
import torch
import wandb

from grid import make_grid, log_px
from objectives import elbo_ffg
from utils import load_dataset, load_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

# parameters
p = argparse.ArgumentParser()
p.add_argument("--idx", type=int, default=0)   
p.add_argument("--hidden_dims", type=int, nargs="+", default=[100])
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--sample_size", type=int, default=64)
p.add_argument("--split", type=str, default="val")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

torch.manual_seed(args.seed)

name_vae = "vae_ffg_h100_tanh_seed0"               # decoder of the Cremer et al. setting
run_name = f"optffg_{name_vae}_{args.split}{args.idx}_seed{args.seed}"
config = vars(args) | {"model": "optimal_ffg", "vae": name_vae}
run = wandb.init(project=args.project, name=run_name, config=config)

# data and frozen decoder
x = load_dataset("mnist_bin")[args.split][args.idx].to(device)        # (784,)
_, dec = load_vae(name_vae, args.hidden_dims, args.latent_dim, "tanh", 784)
dec = dec.to(device).requires_grad_(False)        # gradients flow through the decoder, not into it

# free variational parameters of this image, initialized at the prior N(0, I)
mu = torch.zeros(args.latent_dim, device=device, requires_grad=True)
logsig = torch.zeros(args.latent_dim, device=device, requires_grad=True)
optimizer = torch.optim.Adam([mu, logsig], lr=args.lr)

for step in range(args.steps):
    loss = -elbo_ffg(x, dec, mu, logsig, args.sample_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        wandb.log({"neg_elbo": loss.item()}, step=step)

with torch.no_grad():
    final_elbo = torch.stack([elbo_ffg(x, dec, mu, logsig, S=1000) for _ in range(20)]).mean().item()
    Z, log_prior, log_dA = make_grid(4.0, 400, device=device)
    lpx = log_px(x[None], Z, dec, log_prior, log_dA).item()

run.summary["final_elbo"] = final_elbo
run.summary["log_px"] = lpx
run.summary["kl"] = lpx - final_elbo                # KL(q*_FFG || p(z|x)) = log p(x) - ELBO

path = f"/kaggle/working/{run_name}.pt"
torch.save({"idx": args.idx, "mu": mu.detach().cpu(), "logsig": logsig.detach().cpu()}, path)
art = wandb.Artifact(name=run_name, type="ffg", metadata=config | {"final_elbo": final_elbo})
art.add_file(path)
run.log_artifact(art)
run.finish()