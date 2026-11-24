"""Step A of the training of FVI shared flow: this step aims to train the flow, in order to find its optimise parameters
To do this, the training phase A jointly optimise nu and the flow parameters with annealing. In the step B, the 
nu params (FVI params) are thrown away and trained in FVI with hot start using annealing."""
import argparse

import torch
import wandb

from model import Flow2D
from objectives import elbo_shared_flow
from utils import load_dataset, load_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default="morph_val0_1_N1000")
p.add_argument("--n_layers", type=int, default=8)              # amount of layers for the flow 
p.add_argument("--hidden_dim_flow", type=int, default=32)      #hidden dimension of the flow
p.add_argument("--steps_T", type=int, default=5000)            # step A: learning the shared flow 
                                                               #(learns the parameters of the flow with the parameters nu at the same time)

p.add_argument("--lr_T", type=float, default=1e-3)
p.add_argument("--weight", type=float, default=3.0)            # weight of images with alpha in [0.15, 0.6]
p.add_argument("--lr_nu", type=float, default=1e-2)
p.add_argument("--beta0", type=float, default=0.01)            # likelihood tempering during step A
p.add_argument("--anneal_frac_A", type = float, default = 0.5) # fraction of the train where there is annealing. 
                                                               # For a value 0.5, the annealing stops at the mid of the training
p.add_argument("--sample_size", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

torch.manual_seed(args.seed)
name_vae = "vae_ffg_h100_tanh_seed0"
run_name = f"optsharedflow_{args.data}_L{args.n_layers}_h{args.hidden_dim_flow}_annA{args.anneal_frac_A}_seed{args.seed}"
config = vars(args) | {"model": "fvi_shared_flow", "vae": name_vae}
run = wandb.init(project=args.project, name=run_name, config=config)

data = load_dataset(args.data)
X = data["X"].to(device)                                       # (B, 784)
B = X.shape[0]
_, dec = load_vae(name_vae, hidden_dim_enc=[100], hidden_dim_dec=[100], latent_dim=2, act="tanh", input_dim=784)
dec = dec.to(device).requires_grad_(False)

# step A: the shared flow T and one base per image, optimized together
T = Flow2D(n_layers=args.n_layers, hidden=args.hidden_dim_flow).to(device)
layers = T.layers                                              # only the coupling layers are used; T's own base is ignored
mu = torch.randn(B, 2, device=device).requires_grad_(True)     # starts drawn from the prior
logsig = torch.zeros(B, 2, device=device, requires_grad=True)
opt = torch.optim.Adam([{"params": layers.parameters(), "lr": args.lr_T}, {"params": [mu, logsig], "lr": args.lr_nu}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps_T, eta_min=1e-5)

w = torch.ones(B, device=device)
w[(data["alpha"] >= 0.15) & (data["alpha"] <= 0.6)] = args.weight #weights on the images

#each image with alpha in [0.16, 0.6] has 3 times more weight than others

for step in range(args.steps_T):
    beta = min(1.0, args.beta0 + (1 - args.beta0) * step / (args.anneal_frac_A * args.steps_T))   # ramp to 1 over anneal_frac_A*steps steps 
    elbo = elbo_shared_flow(x=X, dec=dec, layers=layers, mu=mu, logsig=logsig, S=args.sample_size, beta=beta)
    loss = -(w * elbo).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 10 == 0:
        wandb.log({"stepA_neg_elbo_annealed": -elbo.mean().item(), "beta": beta}, step=step)

path = f"/kaggle/working/{run_name}.pt"
torch.save({"alpha": data["alpha"], "mu": mu.detach().cpu(), "logsig": logsig.detach().cpu(),
            "flow_state": {k: v.cpu() for k, v in T.state_dict().items()},
            "n_layers": args.n_layers, "hidden": args.hidden_dim_flow}, path)

art = wandb.Artifact(name=run_name, type="sharedflow", metadata=config)
art.add_file(path)
run.log_artifact(art)
run.finish()