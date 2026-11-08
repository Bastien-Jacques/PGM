import os
import torch
import torch.nn as nn
import torch.nn.functional as F


## This Python files contains the VAE model used by Cremer et al for their Figure 2. The architecture used is:

#encoder: 784 - 100 - 4 = (mean_1, mean_2, logsigma_1, logsigma_2): 
#the encoder produces an output of size 2*d where d is the latent 
#vector size since there is two parameters (mean and variance to produce for each dimension)

#decoder: 2 - 100 - 784  the decoder takes an input z = (z1, z2) 

#Activations are tanh, batch size B = 50, epcohs = 300, lr = 1e-4 using ADAM optimizer

device = "cuda" if torch.cuda.is_available() else "cpu" #for gpu use 

## Encoder ##

activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}

def make_mlp(in_dim: int, hidden_dims: list, act: str):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(activations[act]())     
        d = h                                
    return nn.Sequential(*layers)

class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act = 'tanh'):
        super().__init__()
        self.trunk = make_mlp(input_dim, hidden_dims, act)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logsigma = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.trunk(x)
        mu, logsigma = self.mu(h), self.logsigma(h)
        return mu, logsigma


## Decoder ##

class decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, act = 'tanh'):
        super().__init__()
        self.trunk = make_mlp(latent_dim, hidden_dims, act)
        self.logits = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.trunk(z)
        logits = self.logits(h)

        return logits     #(B, 784)

## function to load vae with the weights

def load_vae(artifact_dir, run_name, hidden_dims, latent_dim=2, act="tanh", input_dim=784):
    """Rebuild encoder and decoder with the given architecture and load trained weights on CPU."""
    enc = encoder(input_dim, hidden_dims, latent_dim, act)
    dec = decoder(latent_dim, hidden_dims, input_dim, act)
    enc.load_state_dict(torch.load(os.path.join(artifact_dir, f"{run_name}_enc_vae.pt"), map_location="cpu"))
    dec.load_state_dict(torch.load(os.path.join(artifact_dir, f"{run_name}_dec_vae.pt"), map_location="cpu"))
    return enc.eval(), dec.eval()