"""Continuation along the morphing path for the shared-flow F-VI: T frozen, each base warm-started from the previous image, it is the step B of the training of the FVI flow
with a hot start, meaning the optimisation of the FVI parameter nu_n starts at nu_n-1  """
import argparse

import torch
import wandb

from model import Flow2D
from objectives import elbo_shared_flow
from utils import load_dataset, load_run, load_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--flow_run", type=str, default="optsharedflow_morph_val0_1_N1000_L8_h32_K10_seed0")
p.add_argument("--data", type=str, default="morph_val0_1_N1000")
p.add_argument("--steps", type=int, default=300)              # optimization steps per image
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--sample_size", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

torch.manual_seed(args.seed)
name_vae = "vae_ffg_h100_tanh_seed0"
run_name = f"cont_{args.flow_run}_steps{args.steps}"
run = wandb.init(project=args.project, name=run_name, config=vars(args) | {"model": "fvi_shared_flow_continuation"})

# same data, decoder and frozen flow as the random-restart run
data = load_dataset(args.data)
X = data["X"].to(device)
B = X.shape[0]
_, dec = load_vae(name_vae, hidden_dim_enc=[100], hidden_dim_dec=[100], latent_dim=2, act="tanh", input_dim=784)
dec = dec.to(device).requires_grad_(False)

prev = load_run(args.flow_run)
T = Flow2D(n_layers=prev["n_layers"], hidden=prev["hidden"])
T.load_state_dict(prev["flow_state"])
layers = T.layers.to(device).requires_grad_(False)

# best random restart of each image, the reference to beat
nu_all = torch.cat([prev["mu_all"], prev["logsig_all"]], dim=2)          # (K, B, 4)
nu_best = nu_all[prev["elbo_all"].argmax(0), torch.arange(B)]            # (B, 4)

# forward sweep (image n) and backward sweep (image B-1-n) optimized together as a batch of 2
nu_fwd, nu_bwd = torch.empty(B, 4), torch.empty(B, 4)
nu = torch.stack([nu_best[0], nu_best[-1]]).to(device)                   # (2, 4)
for n in range(B):
    ids = [n, B - 1 - n]
    mu = nu[:, :2].clone().requires_grad_(True)
    logsig = nu[:, 2:].clone().requires_grad_(True)
    opt = torch.optim.Adam([mu, logsig], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    for _ in range(args.steps):
        loss = -elbo_shared_flow(X[ids], dec, layers, mu, logsig, args.sample_size).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    nu = torch.cat([mu, logsig], dim=1).detach()
    nu_fwd[n], nu_bwd[B - 1 - n] = nu[0].cpu(), nu[1].cpu()
    if n % 50 == 0:
        wandb.log({"image": n, "neg_elbo_fwd": loss.item()})


@torch.no_grad()
def evaluate(nu):
    """ELBO of each image at the given bases, with common random numbers across the three solutions."""
    torch.manual_seed(1234)
    nu = nu.to(device)
    return torch.stack([elbo_shared_flow(X, dec, layers, nu[:, :2], nu[:, 2:], S=50) for _ in range(10)]).mean(0).cpu()


elbo_best, elbo_fwd, elbo_bwd = evaluate(nu_best), evaluate(nu_fwd), evaluate(nu_bwd)
print(f"mean ELBO: best restart {elbo_best.mean():.3f}, forward {elbo_fwd.mean():.3f}, backward {elbo_bwd.mean():.3f}")

path = f"/kaggle/working/{run_name}.pt"
torch.save({"alpha": data["alpha"], "nu_best": nu_best, "nu_fwd": nu_fwd, "nu_bwd": nu_bwd,
            "elbo_best": elbo_best, "elbo_fwd": elbo_fwd, "elbo_bwd": elbo_bwd}, path)
art = wandb.Artifact(name=run_name, type="sharedflow_continuation", metadata=vars(args))
art.add_file(path)
run.log_artifact(art)
run.finish()