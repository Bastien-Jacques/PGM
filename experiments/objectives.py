import math as m
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence


def elbo(x, enc, dec, family = 'ffg', reduce = True):  #Used for the taining in VAE mode with FFG
    mu, logsigma = enc(x)
    z = mu + torch.exp(logsigma)*torch.randn_like(mu)
    logits = dec(z)
    reconstruction = -F.binary_cross_entropy_with_logits(logits, x, reduction = "none").sum(-1) 
    # BCE  = -xlog(p) - (1-x)log(1-p) >0 
    # with_logits means the sigmoid is directly calculated in the loss
    # sum -1: we sum on the 784 pixels. reconstruction = (B,)

    if family != "ffg":
        raise NotImplementedError(f"famille {family} pas encore implémentée")

    q = Independent(Normal(mu, torch.exp(logsigma)), 1)  # approximate posterior q(z|x) with FFG = diagonal gaussian
    prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(mu)), 1) # prior = N(0,I)

    kl = kl_divergence(q, prior) #(B,)

    elbo = reconstruction - kl # training minimizes - ELBO

    return elbo.mean() if reduce else elbo #returns the average loss on a batch if redude, else the elbo per data 

def elbo_ffg(X, dec, mu, logsig, S = 64):
    #X of shape (B, 784) 
    B = X.shape[0]
    eps = torch.randn(S, B, 2, device = mu.device)
    Z = mu + torch.exp(logsig)*eps #(S, B, 2)
    Z = Z.reshape(-1, 2)
    logits = dec(Z.reshape(-1, 2)).reshape(S, B, -1)                                 # (S, B, 784)
    log_lik = (X * F.logsigmoid(logits) + (1 - X) * F.logsigmoid(-logits)).sum(-1)   # (S, B)
    log_lik = log_lik.mean(0)                                                        # average over samples, (B,)
    kl = (0.5 * (mu ** 2 + torch.exp(2 * logsig) - 1) - logsig).sum(-1)              # (B,)

    return log_lik - kl

def elbo_ffg(x, dec, mu, logsig, S=64):
    # x is a single binarized image, shape (784,); mu, logsig are the free variational parameters, shape (2,)
    z = mu + torch.exp(logsig) * torch.randn(S, 2, device=mu.device)                 # (S, 2)
    logits = dec(z)                                                                  # (S, 784)
    log_lik = x @ F.logsigmoid(logits).T + (1 - x) @ F.logsigmoid(-logits).T         # (S,)
    kl = (0.5 * (mu ** 2 + torch.exp(2 * logsig) - 1) - logsig).sum()                # closed-form KL to the prior
    return log_lik.mean() - kl
