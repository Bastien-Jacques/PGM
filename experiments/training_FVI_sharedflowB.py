"""Continuation along the morphing path for the shared-flow F-VI: T frozen, each base warm-started
from the previous image. This is step B of the shared-flow F-VI, with a hot start: the optimization
of nu_n starts at nu_{n-1}, with likelihood tempering inside each image."""
import argparse

import torch
import wandb

from model import Flow2D
from objectives import elbo_shared_flow
from utils import load_dataset, load_run, load_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--flow_run", type=str, default="optsharedflow_morph_val0_1_N1000_L8_h32_annA0.7_seed0")
p.add_argument("--data", type=str, default="morph_val0_1_N1000")
p.add_argument("--steps", type=int, default=500)              # optimization steps per image
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--sample_size", type=int, default=32)
p.add_argument("--beta0", type=float, default=0.01)           # likelihood tempering inside each image
p.add_argument("--anneal_frac_B", type=float, default=0.5)    # fraction of the steps of each image over which beta ramps to 1
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

torch.manual_seed(args.seed)
name_vae = "vae_ffg_h100_tanh_seed0"
run_name = f"cont_{args.flow_run}_steps{args.steps}_annB{args.anneal_frac_B}"
run = wandb.init(project=args.project, name=run_name, config=vars(args) | {"model": "fvi_shared_flow_continuation"})

# same data, decoder and frozen flow as step A
data = load_dataset(args.data)
X = data["X"].to(device)
B = X.shape[0]
_, dec = load_vae(name_vae, hidden_dim_enc=[100], hidden_dim_dec=[100], latent_dim=2, act="tanh", input_dim=784)
dec = dec.to(device).requires_grad_(False)

prev = load_run(args.flow_run)
T = Flow2D(n_layers=prev["n_layers"], hidden=prev["hidden"])
T.load_state_dict(prev["flow_state"])
layers = T.layers.to(device).requires_grad_(False)
nu_init = torch.cat([prev["mu"], prev["logsig"]], dim=1)                 # step A bases, used only to start the sweeps

# forward sweep (image n) and backward sweep (image B-1-n) optimized together as a batch of 2
nu_fwd, nu_bwd = torch.empty(B, 4), torch.empty(B, 4)
nu = torch.stack([nu_init[0], nu_init[-1]]).to(device)                   # (2, 4)
for n in range(B):
    ids = [n, B - 1 - n]
    mu = nu[:, :2].clone().requires_grad_(True)
    logsig = nu[:, 2:].clone().requires_grad_(True)
    opt = torch.optim.Adam([mu, logsig], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    for step in range(args.steps):
        beta = min(1.0, args.beta0 + (1 - args.beta0) * step / (args.anneal_frac_B * args.steps))
        loss = -elbo_shared_flow(X[ids], dec, layers, mu, logsig, args.sample_size, beta).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    nu = torch.cat([mu, logsig], dim=1).detach()
    nu_fwd[n], nu_bwd[B - 1 - n] = nu[0].cpu(), nu[1].cpu()
    if n % 50 == 0:
        with torch.no_grad():
            e = elbo_shared_flow(X[ids], dec, layers, mu, logsig, S=200)
        wandb.log({"image": n, "neg_elbo_fwd": -e[0].item(), "neg_elbo_bwd": -e[1].item()})


@torch.no_grad()
def evaluate(nu):
    """ELBO of each image at the given bases, with common random numbers across the two sweeps."""
    torch.manual_seed(1234)
    nu = nu.to(device)
    return torch.stack([elbo_shared_flow(X, dec, layers, nu[:, :2], nu[:, 2:], S=50) for _ in range(10)]).mean(0).cpu()


elbo_fwd, elbo_bwd = evaluate(nu_fwd),