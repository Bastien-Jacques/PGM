import os
import math as m
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

class Encoder(nn.Module):
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

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, act = 'tanh'):
        super().__init__()
        self.trunk = make_mlp(latent_dim, hidden_dims, act)
        self.logits = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.trunk(z)
        logits = self.logits(h)

        return logits     #(B, 784)


# z_1'=z_1,   z_2'=z_2 * e^{s(z_1)} + t(z_1) where s and t are learned by the model
# log det |\partial z' /partial z| = s(z_1)

# RealNVPCoupling makes the coupling between the two dimesnions of z,the forward function computes the new latent vector z_new given the original z.
# the function inverse gives the original z given z_new (inverses the coupling): not used for the optimisation but for the plot

class RealNVPCoupling(nn.Module):
    def __init__(self, hidden = 32, flip = False):
        super().__init__()
        self.flip = flip #if True, the second coordinate remains unchanged while the first is modfied 
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 2))  #takes one coordinate in input and gives s and t as outputs
        nn.init.zeros_(self.net[-1].weight) #initialisation of the net weights to zero
        nn.init.zeros_(self.net[-1].bias)    #initialisation of the net bias to zero
        # s = t = 0 at start: the layer starts as the identity

    def forward(self, z):
        a, b = (z[:, 0:1], z[:, 1:2]) if not self.flip else (z[:, 1:2], z[:, 0:1])
        s, t = self.net(a).chunk(2, dim = -1)  #(S,1) S is the amount of latent vectors sampled in one shot from the flow
        b_new = b*torch.exp(s) + t 
        z_new = torch.cat([a, b_new], dim = -1) if not self.flip else torch.cat([b_new, a], dim = -1)
        log_det = s.squeeze(-1)  #(S,)

        return z_new, log_det

    def inverse(self, z_new):
        a_new, b_new = (z_new[:, 0:1], z_new[:, 1:2]) if not self.flip else (z_new[:, 1:2], z_new[:, 0:1])
        s, t = self.net(a_new).chunk(2, dim = -1)  #(S,1) 

        a, b = a_new, (b_new - t)*torch.exp(-s)
        z = torch.cat([a,b], dim = -1) if not self.flip else torch.cat([b,a], dim = -1)
        log_det = -s.squeeze(-1)

        return z, log_det



#creates the flow. From z0 ~ q0 = N(mu, diag(sigma^2)), we apply K NVP coupling with alternative true and false flip
# zK = f_K ° .... ° f1(z0)
# variable changement formula gives log q(zK) = log q0 (z0) - sum_{k=1}^K log | det \partial f_k / \partial z |

class Flow2D(nn.Module):
    def __init__(self, n_layers = 8, hidden = 32, mu_init = None, logsigma_init = None):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(2) if mu_init is None else mu_init.clone())
        self.logsigma = nn.Parameter(torch.zeros(2) if logsigma_init is None else logsigma_init.clone())
        self.layers = nn.ModuleList([RealNVPCoupling(hidden, flip = (k%2 ==1)) for k in range(n_layers)]) #the layer is pair = no flip, else flip

    def log_q0(self, z0):
    #Log density of the Gaussian base at z0 (S, 2)
        u = (z0 - self.mu) / torch.exp(self.logsigma)
        return -0.5 * (u ** 2).sum(-1) - self.logsigma.sum() - m.log(2 * m.pi)

    def sample(self, S):
        # S reparameterised samples z_K and their log density log q(z_K)
        z = self.mu + torch.exp(self.logsigma) * torch.randn(S, 2)
        log_q = self.log_q0(z)
        for layer in self.layers:
            z, log_det = layer(z)
            log_q = log_q - log_det
        return z, log_q

    def log_prob(self, z):
        #Log density of the flow at given points z (S, 2), through the inverse layers
        log_det_sum = torch.zeros(z.shape[0])
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)  #invert the layer and accumulate its log-det
            log_det_sum += log_det

        return self.log_q0(z) + log_det_sum 
    
## function to load vae with the weights

def load_vae(artifact_dir, run_name, hidden_dims, latent_dim=2, act="tanh", input_dim=784):
    """Rebuild encoder and decoder with the given architecture and load trained weights on CPU."""
    enc = Encoder(input_dim, hidden_dims, latent_dim, act)
    dec = Decoder(latent_dim, hidden_dims, input_dim, act)
    enc.load_state_dict(torch.load(os.path.join(artifact_dir, f"{run_name}_enc_vae.pt"), map_location="cpu"))
    dec.load_state_dict(torch.load(os.path.join(artifact_dir, f"{run_name}_dec_vae.pt"), map_location="cpu"))
    return enc.eval(), dec.eval()