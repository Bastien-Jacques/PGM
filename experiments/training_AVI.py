"""A-VI with a frozen decoder: train a new encoder on a dataset artifact (MNIST or a morphing path),
in the FFG family or in the shared-flow family q = T# N(mu, sigma^2) with T frozen."""
import argparse

import torch
import wandb

from model import Encoder, Flow2D
from objectives import elbo, elbo_shared_flow
from utils import load_dataset, load_encoder, load_run, load_vae

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
p.add_argument("--init_from_enc", type=str, default=None)        # name of a saved encoder run to start from (same architecture)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--family", type=str, default="ffg", choices=["ffg", "sharedflow"])
p.add_argument("--flow_run", type=str, default=None)             # step A run with the frozen T (shared-flow family)
p.add_argument("--cont_run", type=str, default=None)             # continuation run: F-VI optimum in the same family
p.add_argument("--flow_tag", type=str, default="w5")             # short tag of the flow, for the run name
p.add_argument("--sample_size", type=int, default=32)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

name_vae = "vae_ffg_h100_tanh_seed0"               # reference VAE: only its decoder is used, and it stays frozen
init_tag = "_ft" if args.init_from_enc is not None else ""          # fine-tuned from a pre-trained encoder
family_tag = args.family if args.family == "ffg" else f"sharedflow-{args.flow_tag}"
run_name = (f"enc_{args.data}_{family_tag}_h{'-'.join(map(str, args.hidden_dim_enc))}_{args.act}"
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

# shared-flow family: the frozen T learned by F-VI on the path
if args.family == "sharedflow":
    prev = load_run(args.flow_run)
    T = Flow2D(n_layers=prev["n_layers"], hidden=prev["hidden"])
    T.load_state_dict(prev["flow_state"])
    layers = T.layers.to(device).requires_grad_(False)


def neg_elbo(x_batch):
    """Mean negative ELBO over the batch, in the chosen family."""
    if args.family == "ffg":
        return -elbo(x_batch, enc, dec)
    mu, logsig = enc(x_batch)
    return -elbo_shared_flow(x_batch, dec, layers, mu, logsig, args.sample_size).mean()


optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=50)
enc_path = f"/kaggle/working/{run_name}_enc.pt"
best_val = float("inf")
best_train_loss = float("inf")

for epoch in range(args.epochs):
    # one pass over the data in a random order
    train_loss_tot = 0.0
    perm = torch.randperm(N, device=device)
    for i in range(0, N, args.batch_size):
        x_batch = X[perm[i:i + args.batch_size]]
        loss = neg_elbo(x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_tot += loss.item() * x_batch.shape[0]
    scheduler.step(train_loss_tot)
    lr_now = optimizer.param_groups[0]["lr"]
    log = {"epoch": epoch, "train_loss": train_loss_tot / N, "lr": lr_now}

    # morphing path: keep the encoder with the best training loss
    if X_val is None and train_loss_tot < best_train_loss:
        best_train_loss = train_loss_tot
        torch.save(enc.state_dict(), enc_path)

    # MNIST: keep the encoder with the best validation loss
    if X_val is not None:
        with torch.no_grad():
            val_loss = neg_elbo(X_val).item()
        log["val_loss"] = val_loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(enc.state_dict(), enc_path)

    wandb.log(log)

meta = config | ({"best_validation": best_val} if X_val is not None else {"best_train_loss": best_train_loss / N})
art = wandb.Artifact(name=run_name, type="encoder", metadata=meta)
art.add_file(enc_path)

# shared flow on the path: amortization gap and stiffness of the best encoder, same estimator as the continuation
if args.family == "sharedflow" and X_val is None and args.cont_run is not None:
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    with torch.no_grad():
        mu, logsig = enc(X)
        nu = torch.cat([mu, logsig], dim=1)
        torch.manual_seed(1234)
        elbo_A = torch.stack([elbo_shared_flow(X, dec, layers, mu, logsig, S=50) for _ in range(10)]).mean(0).cpu()
        kappa = ((nu[1:] - nu[:-1]).norm(dim=1) / (X[1:] - X[:-1]).norm(dim=1)).max().item()
    gap = load_run(args.cont_run)["elbo_fwd"] - elbo_A
    print(f"kappa_hat = {kappa:.1f}, mean gap = {gap.mean():.3f} nats, largest gap = {gap.max():.2f} nats "
          f"at alpha = {data['alpha'][gap.argmax()]:.3f}")
    run.summary.update({"kappa_hat": kappa, "mean_gap": gap.mean().item(), "max_gap": gap.max().item()})
    res_path = f"/kaggle/working/{run_name}_eval.pt"
    torch.save({"alpha": data["alpha"], "nu": nu.cpu(), "elbo_A": elbo_A, "gap": gap, "kappa": kappa}, res_path)
    art.add_file(res_path)

run.log_artifact(art)
run.finish()