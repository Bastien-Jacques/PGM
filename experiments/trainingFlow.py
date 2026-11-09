import torch
import os
from model import Encoder, Decoder, Flow2D
from utils import load_artifact, load_vae, load_dataset
from grid import make_grid, log_px
from objectives import elbo_flow_mc
import wandb 
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#parameters

p = argparse.ArgumentParser()

p.add_argument("--hidden_dim_flow", type=int, default=32)  
p.add_argument("--n_layers", type=int, default=8) 
p.add_argument("--hidden_dims", type=int, nargs="+", default=[100])  
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--sample_size", type=int, default=50)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split", type=str, default="val")
p.add_argument("--idx", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
p.add_argument("--annealing",  action = 'store_true')
p.add_argument("--beta0", type = float, default = 0)
args = p.parse_args()

torch.manual_seed(args.seed)

name_vae = "vae_ffg_h100_tanh_seed0" #run with the Cremer et al decoder

anneal_tag = f"_anneal{args.beta0}" if args.annealing else ""
run_name = f"optflow_L{args.n_layers}_h{args.hidden_dim_flow}{anneal_tag}_{name_vae}_{args.split}{args.idx}_seed{args.seed}"

config = vars(args) | {"model": "optimal_flow"}   
run = wandb.init(project=args.project, name=run_name, config=config)

data = load_dataset("mnist_bin")
X = data[args.split].to(device)

N = X.shape[0] #amount of data 
S = args.sample_size  #sample size

hidden_dim_flow = args.hidden_dim_flow
n_layers = args.n_layers
hidden_dims = args.hidden_dims
lr = args.lr
steps = args.steps


## Training for F-VI with flow. Strating from z ~p(z) = N(0,I)
flow = Flow2D(n_layers = n_layers, hidden = hidden_dim_flow).to(device)
optimizer = torch.optim.Adam(list(flow.parameters()), lr = lr) ## Important, we only train flow network, not decoder which is already trained

_, dec = load_vae(name_vae, hidden_dims, latent_dim=2, act="tanh", input_dim=784)
dec = dec.to(device).requires_grad_(False) #gradient not needed since the decoder is already trained: frozen decoder

x = X[args.idx]
x = x.to(device)

def beta_schedule(beta0, step, steps, annealing = None): #beta annealing
    if not annealing:
        return 1
    
    else: return min(beta0 + 2*step/steps, 1)


for step in range(steps):
    beta = beta_schedule(args.beta0, step, steps, args.annealing)
    loss = -elbo_flow_mc(x, dec, flow, S, beta) #scalar, - to minimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%10 ==0: 
        with torch.no_grad():
            loss_follow = -elbo_flow_mc(x, dec, flow, S, beta = 1) #scalar, - to minimize
        wandb.log({"neg_elbo": loss_follow.item(), "neg_elbo_annealed": loss.item(), "beta": beta}, step = step) 
        #log the loss every 10 steps


    # final evaluation with many samples, so the reported ELBO is not noisy
with torch.no_grad():
    final_elbo = elbo_flow_mc(x, dec, flow, 20_000).item()
    Z, log_prior, log_dA = make_grid(4.0, 400, device=device)
    lpx = log_px(x[None], Z, dec, log_prior, log_dA).item()   # batch of one image -> float

run.summary["final_elbo"] = final_elbo #final elbo (after n_steps steps)
run.summary["log_px"] = lpx            #log evidence log p(x) after n_steps steps
run.summary["kl_flow"] = lpx - final_elbo # KL(q_flow || p(z|x)) = log p(x) - ELBO


path = f"/kaggle/working/{run_name}.pt"
torch.save(flow.state_dict(), path)
art = wandb.Artifact(name=run_name, type="flow", metadata=config | {"final_elbo": final_elbo})
art.add_file(path)
run.log_artifact(art)


run.finish() # wandb end
