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

class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, latent_dim)
        self.l3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        return self.l2(h), self.l3(h)


## Decoder ##

class decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, outpout_dim):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, outpout_dim)

    def forward(self, z):
        h = torch.tanh(self.l1(z))
        output = self.l2(h)

        return output     #(B, 784)


