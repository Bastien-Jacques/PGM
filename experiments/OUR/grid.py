## This code enables the compute of the true posterior and likelihood for a 2D latent space

import math as m
import torch
import torch.nn.functional as F

from model import encoder, decoder
from objectives import ELBO

enc = encoder(784, 100, 2)
dec = decoder(2, 100, 784)

def make_grid(L, n, device = 'cpu'):
    g = torch.linspace(-L, L, n, device = device) #one segment between -L and L with n points
    Z = torch.cartesian_prod(g,g)                 #cartesian product of 2 segments = [-L,L]^2: grid of n*n squares
    log_prior = -0.5 * (Z ** 2).sum(-1) - m.log(2 * m.pi)      # (K,)  #p(z) ~N(0,I) (K,)
    h = 2*L/(n-1) #discretisation size
    log_dA = 2 * m.log(h)        # log(h^2) = log of the surfaces in the grid. The area of one square is h*h = h^2  (K,)

    return Z, log_prior, log_dA

@torch.no_grad()
def log_joint_grid(X, Z, dec, log_prior):
    logits = dec(Z) # (K, 784) K = n^2 = amount of vectors z on the grid 

    #for a given data x_b (784,) with x_b,j the j pixel of the image b, and a latent vector z_k = (2,) in the grid, the log likelihood can be computed as:
    #log p(x_b | z_k) = Sum_j x_b,j log(sigmoide(dec(z_k))_j) + (1 - x_b,j)log(1 - sigmoide(dec(z_k))_j) = x_b.log(dec)
    log_likelihood =  X @ F.logsigmoid(logits).T + (1 - X) @ F.logsigmoid(-logits).T   # (B, K)  log p(x_b | z_k) 

    return log_prior + log_likelihood   #(B, K)  log p(z_k) + log p(x_b | z_k) = log p(x, z)

@torch.no_grad()
def log_px(X, Z, dec, log_prior, log_dA):
    return torch.logsumexp(log_joint_grid(X, Z, dec, log_prior), dim=1) + log_dA     # (B,)

@torch.no_grad()
def posterior_on_grid(X, Z, dec, log_prior, log_dA):
    lj = log_joint_grid(X, Z, dec, log_prior)                     # (B, K)
    lpx = torch.logsumexp(lj, dim=1, keepdim=True) + log_dA       # (B, 1)
    return torch.exp(lj - lpx)                                    # (B, K), obtained by Bayes


device = 'cpu'

data = torch.load("mnist_bin.pt")
X_val = data['val']

Z, log_prior, log_dA = make_grid(4.0, 200, device)
post = posterior_on_grid(X_val[:5], Z, dec, log_prior, log_dA)

L = 4
n = 200
Z, lp, lda = make_grid(L, n, device)
print(L, n, log_px(X_val[:1000], Z, dec, lp, lda).mean().item())

print("ELBO", ELBO(X_val[:1000], enc, dec).item())     