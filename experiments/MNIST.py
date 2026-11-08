import numpy as np
import os
import torch
from torchvision import datasets

g = torch.Generator().manual_seed(0) #fixed seed

train_raw  = datasets.MNIST("data", train = True, download = True)
test_raw   = datasets.MNIST("data", train = False, download = True)

X = train_raw.data.float().div(255).view(-1, 784) # 60000 x 784, velues in [0, 1]: Train data 
#.view(-1, 784) flattens the (60_000,28,28) tensor into a (60000,784) vector

X_te = test_raw.data.float().div(255).view(-1, 784)  # 10000 x 784 : test data 

X_bin = torch.bernoulli(X, generator = g)      # each pixel is either 0 or 1. P(pixel = 1) = intensity
X_te_bin = torch.bernoulli(X_te, generator = g)

perm = torch.randperm(np.shape(X)[0], generator = g) #random permutation of interval [0,59999]

train_size = 50_000 #size of the validation set 

X_train, X_val = X_bin[perm[0:train_size]], X_bin[perm[train_size:]] #training set and validation set


OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "."   # Kaggle ou local
torch.save({"train": X_train, "val": X_val, "test": X_te_bin},
           os.path.join(OUT_DIR, "mnist_bin.pt"))