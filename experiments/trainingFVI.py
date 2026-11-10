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
run_name = f"optffg_{name_vae}_{args.split}_{args.idx}_seed{args.seed}"
config = vars(args) | {"model": "optimal_ffg", "vae": name_vae}
run = wandb.init(project=args.project, name=run_name, config=config)

# data and frozen decoder
X = load_dataset("mnist_bin")[args.split][args.idxs].to(device)          # (B, 784)
B = X.shape[0]
_, dec = load_vae(name_vae, args.hidden_dims, args.latent_dim, "tanh", 784)
dec = dec.to(device).requires_grad_(False)          # gradients flow through the decoder, not into it

# free variational parameters, one row per image, initialized at the prior N(0, I)
mu = torch.zeros(B, args.latent_dim, device=device, requires_grad=True)
logsig = torch.zeros(B, args.latent_dim, device=device, requires_grad=True)
optimizer = torch.optim.Adam([mu, logsig], lr=args.lr)

for step in range(args.steps):
    elbo = elbo_ffg(X, dec, mu, logsig, args.sample_size)                 # (B,)
    loss = -elbo.sum()                              # images are independent: sum, so the step size does not depend on B
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        wandb.log({"neg_elbo": -elbo.mean().item()}, step=step)

# final evaluation: average of independent Monte Carlo estimates
with torch.no_grad():
    final_elbo = torch.stack([elbo_ffg(X, dec, mu, logsig, S=1000) for _ in range(20)]).mean(0)   # (B,)
    Z, log_prior, log_dA = make_grid(4.0, 400, device=device)
    lpx = log_px(X, Z, dec, log_prior, log_dA)      # exact log p(x) on the grid, (B,) 
    kl = lpx - final_elbo                           # KL(q*_FFG || p(z|x)) per image 

run.summary["final_elbo"] = final_elbo #final elbo (after n_steps steps)
run.summary["kl"] = lpx - final_elbo # KL(q_flow || p(z|x)) = log p(x) - ELBO


run.summary["kl_mean"] = kl.mean().item()
for idx, k in zip(args.idxs, kl.tolist()):
    run.summary[f"kl_{idx}"] = k

# save everything needed to plot the optimal Gaussians
path = f"/kaggle/working/{run_name}.pt"
torch.save({"idxs": args.idxs, "mu": mu.detach().cpu(), "logsig": logsig.detach().cpu(),
            "elbo": final_elbo.cpu(), "log_px": lpx.cpu()}, path)
art = wandb.Artifact(name=run_name, type="ffg", metadata=config | {"kl_mean": kl.mean().item()})
art.add_file(path)
run.log_artifact(art)

run.finish()