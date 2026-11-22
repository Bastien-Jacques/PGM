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

def elbo_ffg_batch(X, dec, mu, logsig, S=64):
    # X: (B, 784) images; mu, logsig: (B, 2), one row of free variational parameters per image
    eps = torch.randn(S, *mu.shape, device=mu.device)                                 # (S, B, 2)
    z = mu + torch.exp(logsig) * eps                                                 # (S, B, 2)
    logits = dec(z.reshape(-1, 2)).reshape(S, mu.shape[0], -1)                        # (S, B, 784)
    log_lik = (X * F.logsigmoid(logits) + (1 - X) * F.logsigmoid(-logits)).sum(-1)   # (S, B)
    kl = (0.5 * (mu ** 2 + torch.exp(2 * logsig) - 1) - logsig).sum(-1)              # (B,)
    return log_lik.mean(0) - kl                                                      # (B,)

def elbo_ffg(x, dec, mu, logsig, S=64):
    # x is a single binarized image, shape (784,); mu, logsig are the free variational parameters, shape (2,)
    z = mu + torch.exp(logsig) * torch.randn(S, 2, device=mu.device)                 # (S, 2)
    logits = dec(z)                                                                  # (S, 784)
    log_lik = x @ F.logsigmoid(logits).T + (1 - x) @ F.logsigmoid(-logits).T         # (S,)
    kl = (0.5 * (mu ** 2 + torch.exp(2 * logsig) - 1) - logsig).sum()                # closed-form KL to the prior
    return log_lik.mean() - kl

def elbo_flow_mc(x, dec, flow, S=64, beta = 1):
    #x is a single binarized image from the dataset, shape (784,)
    z, log_q = flow.sample(S) #(S,2); (S,)
    logits = dec(z)
    log_lik = x @ F.logsigmoid(logits).T + (1 - x) @ F.logsigmoid(-logits).T   #(S,)  remember that sigma(-y) = 1 - sigma(y) for sigmoid function 
    log_prior = -0.5*(z**2).sum(-1) - m.log(2*m.pi)  #(S,)
    return (beta*log_lik + log_prior - log_q).mean()

def elbo_shared_flow(X, dec, layers, mu, logsig, S=32, beta=1.0):
    # q_n = T # N(mu_n, sigma_n^2), with T the composition of the shared coupling layers (basically as push forward)
    # X: (R, 784) images; mu, logsig: (R, 2) base parameters, one row per image
    eps = torch.randn(S, *mu.shape, device=mu.device)                                  # (S, R, 2)
    z = (mu + torch.exp(logsig) * eps).reshape(-1, 2)                                  # base samples, (S * R, 2)
    log_q = (-0.5 * eps ** 2 - logsig - 0.5 * m.log(2 * m.pi)).sum(-1)                 # base log-density, (S, R)
    for layer in layers:                                                               # push the samples through T
        z, log_det = layer(z)
        log_q = log_q - log_det.view(S, -1)
    z = z.view(S, -1, 2)
    logits = dec(z.reshape(-1, 2)).view(S, z.shape[1], -1)                             # (S, R, 784)
    log_lik = (X * F.logsigmoid(logits) + (1 - X) * F.logsigmoid(-logits)).sum(-1)     # (S, R)
    log_prior = -0.5 * (z ** 2).sum(-1) - m.log(2 * m.pi)                              # (S, R)
    kl = log_q - log_prior 
    return (beta * log_lik -kl).mean(0)                                                # (R,)