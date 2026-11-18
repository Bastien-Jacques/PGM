"""Per-image F-VI on every image of a dataset artifact (e.g. a morphing path), with a FFG or a flow family."""
import argparse
import copy
import os

import torch
import wandb

from model import Flow2D
from objectives import elbo_ffg_batch, elbo_flow_mc
from utils import load_artifact, load_dataset, load_vae, load_flow

device = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--family", type=str, default="ffg", choices=["ffg", "flow"])
p.add_argument("--data", type=str, default="morph_val0_1_N1000")   # dataset artifact holding {"X", "alpha", ...}
p.add_argument('--n_restarts', type = int, default = 10)
p.add_argument("--init", type=str, default="prior")                # "prior", or the name of an optimal FFG / flow run to start from
p.add_argument("--hidden_dims", type=int, nargs="+", default=[100])
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--lr_max", type=float, default=None)               # default: 1e-2 for ffg, 1e-3 for flow
p.add_argument("--lr_min", type=float, default=None)               # default: 1e-2 for ffg, 1e-3 for flow
p.add_argument("--steps", type=int, default=None)                  # default: 3000 for ffg, 1500 for flow
p.add_argument("--sample_size", type=int, default=64)
p.add_argument("--n_layers", type=int, default=8)                  # flow only, used when starting from the prior
p.add_argument("--hidden_dim_flow", type=int, default=32)          # flow only, used when starting from the prior
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

lr_max = args.lr_max if args.lr_max is not None else {"ffg": 1e-2, "flow": 1e-3}[args.family]
lr_min = args.lr_min
steps = args.steps if args.steps is not None else {"ffg": 3000, "flow": 1500}[args.family]
torch.manual_seed(args.seed)

name_vae = "vae_ffg_h100_tanh_seed0"
init_tag = "prior" if args.init == "prior" else args.init.split("_")[-2]     # e.g. "val0" for ..._val0_seed0
run_name = f"opt{args.family}_{args.data}_init{init_tag}_K{args.n_restarts}_seed{args.seed}"
config = vars(args) | {"lr_max": lr_max, "lr_min": lr_min, "steps": steps, "K": args.n_restarts, "model": f"optimal_{args.family}", "vae": name_vae}
run = wandb.init(project=args.project, name=run_name, config=config)

# images of the path and frozen decoder
data = load_dataset(args.data)
X = data["X"].to(device)                                               # (B, 784)
B = X.shape[0]
_, dec = load_vae(name_vae, args.hidden_dims, args.latent_dim, "tanh", 784)
dec = dec.to(device).requires_grad_(False)


def fit_ffg():
    """All images optimized together, every row from a Gaussian of the K different starts"""

    K = args.n_restarts
    XK = X.repeat(K,1) # (K * B, 784): each image K times

    mu = torch.randn(K * B, args.latent_dim, device=device).requires_grad_(True)   # random starts drawn from the prior
    logsig = torch.zeros(K * B, args.latent_dim, device=device, requires_grad=True)

    opt = torch.optim.Adam([mu, logsig], lr=lr_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = steps, eta_min = lr_min)

    for step in range(steps):
        elbo = elbo_ffg_batch(XK, dec, mu, logsig, args.sample_size)   # (K.B, )
        loss = -elbo.sum()                                            # images are independent
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if step % 10 == 0:
            wandb.log({"neg_elbo_per_image": -elbo.mean().item()}, step=step)

    # final ELBO of each image, averaged over independent estimates to keep memory bounded
    with torch.no_grad():
        final_elbo = torch.stack([elbo_ffg_batch(XK, dec, mu, logsig, args.sample_size) for _ in range(50)]).mean(0)
        final_elbo = final_elbo.view(K, B)
        best = final_elbo.argmax(0)                                             # (B,) best restart for each image
        idx = best * B + torch.arange(B, device=device)
    return {"mu": mu.detach()[idx].cpu(), "logsig": logsig.detach()[idx].cpu(),
            "elbo": final_elbo.max(0).values.cpu()}



def fit_flow():
    """One flow per image, each a fresh copy of the starting flow, optimized on its own."""
    if args.init == "prior":
        flow0 = Flow2D(n_layers=args.n_layers, hidden=args.hidden_dim_flow)
    else:
        flow0 = load_flow(args.init, "latest")
    flow0 = flow0.to(device)

    states, final_elbo = [], []
    for i in range(B):
        flow = copy.deepcopy(flow0)                                   # same starting point for every image
        opt = torch.optim.Adam(flow.parameters(), lr=lr_max)
        for step in range(steps):
            loss = -elbo_flow_mc(X[i], dec, flow, args.sample_size)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final_elbo.append(elbo_flow_mc(X[i], dec, flow, 20_000).item())
        states.append({k: v.detach().cpu() for k, v in flow.state_dict().items()})
        wandb.log({"image": i, "final_elbo": final_elbo[-1]})
    return {"states": states, "elbo": torch.tensor(final_elbo),
            "n_layers": len(flow0.layers), "hidden": flow0.layers[0].net[0].out_features}


result = fit_ffg() if args.family == "ffg" else fit_flow()

path = f"/kaggle/working/{run_name}.pt"
torch.save({"alpha": data["alpha"], **result}, path)
art = wandb.Artifact(name=run_name, type=args.family, metadata=config)
art.add_file(path)
run.log_artifact(art)
run.finish()