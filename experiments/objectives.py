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


def elbo_flow_mc(x, dec, flow, S=64):
    #x is a single binarized image from the dataset, shape (784,)
    z, log_q = flow.sample(S) #(S,2); (S,)
    logits = dec(z)
    log_lik = x @ F.logsigmoid(logits).T + (1 - x) @ F.logsigmoid(-logits).T   #(S,)  remember that sigma(-y) = 1 - sigma(y) for sigmoid function 
    log_prior = -0.5*(z**2).sum(-1) - m.log(2*m.pi)  #(S,)
    return (log_lik + log_prior - log_q).mean()
