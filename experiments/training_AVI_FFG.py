"""A-VI with a frozen decoder: train a new FFG encoder on a dataset artifact (MNIST or a morphing path)."""
import argparse

import torch
import torch.nn as nn
import wandb

from model import Encoder
from objectives import elbo
from utils import load_dataset, load_vae, load_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# parameters
p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default="morph_val0_1_N1000")             # "mnist_bin" or a morphing path artifact
p.add_argument("--hidden_dim_enc", type=int, nargs="+", default=[100])       # architecture of the new encoder
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--act", type=str, default="tanh", choices=["tanh", "relu", "elu"])
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=50)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--init_from_enc", type=str, default=None)        # name of a saved encoder run to start from (same architecture
p.add_argument("--seed", type=int, default=0)
p.add_argument("--family", type=str, default="ffg")
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

name_vae = "vae_ffg_h100_tanh_seed0"               # reference VAE: only its decoder is used, and it stays frozen
init_tag = "_ft" if args.init_from_enc is not None else ""          # fine-tuned from a pre-trained encoder
run_name = (f"enc_{args.data}_{args.family}_h{'-'.join(map(str, args.hidden_dim_enc))}_{args.act}"
            f"{init_tag}_dec-{name_vae}_seed{args.seed}")
config = vars(args) | {"model": "avi_frozen_decoder", "vae": name_vae}

torch.manual_seed(args.seed)
run = wandb.init(project=args.project, name=run_name, config=config)

# data: MNIST has a validation set, a morphing path does not (we want the best fit on these very images)
data = load_dataset(args.data)
if args.data == "mnist_bin":
    X, X_val = data["train"].to(device), data["val"].to(device)
elif args.data.startswith("morph"):
    X, X_val = data["X"].to(device), None
else:
    raise ValueError(f"unknown dataset {args.data}")
N, input_dim = X.shape

# new encoder, frozen decoder of the reference VAE
enc = Encoder(input_dim, args.hidden_dim_enc, args.latent_dim, args.act).to(device)
if args.init_from_enc is not None:
    # start from an encoder pre-trained on MNIST with the same frozen decoder
    enc.load_state_dict(load_encoder(args.init_from_enc, args.hidden_dim_enc, latent_dim=args.latent_dim, act=args.act).state_dict())


_, dec = load_vae(name_vae, hidden_dim_enc=[100], hidden_dim_dec=[100],
                  latent_dim=args.latent_dim, act="tanh", input_dim=input_dim)
dec = dec.to(device).requires_grad_(False)        # gradients flow through the decoder, not into it
optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr)

enc_path = f"/kaggle/working/{run_name}_enc.pt"
best_val = float("inf")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience = 50)

for epoch in range(args.epochs):
    # one pass over the data in a random order
    train_loss_tot = 0.0
    perm = torch.randperm(N, device=device)
    for i in range(0, N, args.batch_size):
        x_batch = X[perm[i:i + args.batch_size]]
        loss = -elbo(x_batch, enc, dec)            # mean negative ELBO over the batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_tot += loss.item() * x_batch.shape[0]
    scheduler.step(train_loss_tot)    
    lr_now = optimizer.param_groups[0]["lr"]
    log = {"epoch": epoch, "train_loss": train_loss_tot / N, 'lr':  lr_now}

    # MNIST: keep the encoder with the best validation loss
    if X_val is not None:
        with torch.no_grad():
            val_loss = -elbo(X_val, enc, dec).item()
        log["val_loss"] = val_loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(enc.state_dict(), enc_path)

    wandb.log(log)

# morphing path: no validation, the final encoder is the fit we want
if X_val is None:
    torch.save(enc.state_dict(), enc_path)

meta = config | ({"best_validation": best_val} if X_val is not None else {"final_train_loss": train_loss_tot / N})
art = wandb.Artifact(name=run_name, type="encoder", metadata=meta)
art.add_file(enc_path)
run.log_artifact(art)
run.finish()