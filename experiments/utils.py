import wandb
import torch
import os
from model import Encoder, Decoder, Flow2D

def load_artifact(name, version='v0'):
    api = wandb.Api()
    return api.artifact(f"bastienjacques38-university-/PGM/{name}:{version}").download()

def load_vae(run_name, hidden_dims, latent_dim=2, act="tanh", input_dim=784, version ='v0'):
    #Rebuild encoder and decoder with the given architecture and load trained weights on CPU

    artifact_dir = load_artifact(run_name, version)
    enc = Encoder(input_dim, hidden_dims, latent_dim, act)
    dec = Decoder(latent_dim, hidden_dims, input_dim, act)
    enc.load_state_dict(torch.load(os.path.join(artifact_dir, f"{run_name}_enc_vae.pt"), map_location="cpu"))
    dec.load_state_dict(torch.load(os.path.join(artifact_dir, f"{run_name}_dec_vae.pt"), map_location="cpu"))
    return enc.eval(), dec.eval()

def load_flow(run_name, version = "v0"):
    art = wandb.Api().artifact(f"bastienjacques38-university-/PGM/{run_name}:{version}")
    cfg = art.metadata
    flow = Flow2D(n_layers=cfg["n_layers"], hidden=cfg["hidden_dim_flow"])
    flow.load_state_dict(torch.load(os.path.join(art.download(), f"{run_name}.pt"), map_location="cpu"))
    return flow.eval()

def load_dataset(name="mnist_bin", version="v0"):
    #Download the dataset artifact and load the tensors it contains
    return torch.load(os.path.join(load_artifact(name, version), f"{name}.pt"))