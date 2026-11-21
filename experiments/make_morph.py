"""Build the morphing path x_alpha = (1 - alpha) x_0 + alpha x_N between two images, save it and log it to wandb."""
import argparse

import torch
import wandb

from utils import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--i0", type=int, default=0)             # first endpoint
p.add_argument("--iN", type=int, default=1)             # second endpoint
p.add_argument("--N", type=int, default=1000)           # number of steps: the path has N + 1 images
p.add_argument("--split", type=str, default="val")
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

name = f"morph_{args.split}{args.i0}_{args.iN}_N{args.N}"
run = wandb.init(project=args.project, name=name, job_type="dataset", config=vars(args))

X = load_dataset("mnist_bin")[args.split]
x0, xN = X[args.i0], X[args.iN]

# linear interpolation in pixel space, endpoints included
alpha = torch.linspace(0, 1, args.N + 1)                             # (N + 1,)
X_path = (1 - alpha[:, None]) * x0 + alpha[:, None] * xN             # (N + 1, 784)

# the path must start and end exactly at the two dataset images
assert torch.equal(X_path[0], x0) and torch.equal(X_path[-1], xN)

path = f"{name}.pt"
torch.save({"alpha": alpha, "X": X_path, "i0": args.i0, "iN": args.iN, "split": args.split}, path)

art = wandb.Artifact(name, type="dataset", metadata=vars(args))
art.add_file(path)
run.log_artifact(art)
run.finish()

