import torch
from model import encoder, decoder
from objectives import ELBO
import wandb 

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("PGM_key")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

data = torch.load("/kaggle/working/mnist_bin.pt")
X_train, X_val = data['train'].to(device), data['val'].to(device)

N, B = X_train.shape[0], 50 #amount of training data and batch size


input_dim = X_train.shape[1]
hidden_dim = 100
latent_dim = 2
output_dim = input_dim

enc, dec = encoder(input_dim, hidden_dim, latent_dim ).to(device), decoder(latent_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr = 1e-4)
epochs = 300


config = dict(model="vae", seed=0, hidden_dim=100, latent_dim=2,
              lr=1e-4, batch_size=50, epochs=300, family="FFG")
run = wandb.init(project="encoder-glasses", config=config, name=f"vae_h100_seed0")

best_val = float("inf")  
for epoch in range(epochs):

    train_loss_tot, val_loss_tot = 0, 0
    perm = torch.randperm(N, device = device) #suffle the training data order at each epoch
    
    #training loop
    for i in range(0, N, B): #separation with batches
        x_batch = X_train[perm[i:i+B]]
        train_loss = - ELBO(x_batch, enc, dec) #mean on the batch
        train_loss_tot += train_loss.item()*B #sum of the train loss on the batch
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    #validation loop
    with torch.no_grad():
        val_loss_tot = - ELBO(X_val, enc, dec).item() #average validation loss on the whole validation set
    
    if val_loss_tot < best_val:
        best_val = val_loss_tot
        torch.save(enc.state_dict(), "/kaggle/working/enc_vae.pt")
        torch.save(dec.state_dict(), "/kaggle/working/dec_vae.pt")

    wandb.log({"epoch": epoch, "train_loss": train_loss_tot/N, "val_loss": val_loss_tot})
 

run.finish() # wandb end




