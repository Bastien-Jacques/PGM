import torch
from model import encoder, decoder
from objectives import ELBO
import wandb 
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#parameters

p = argparse.ArgumentParser()
p.add_argument("--hidden_dims", type=int, nargs="+", default=[100])  
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--act", type=str, default="tanh", choices=["tanh", "relu", "elu"])
p.add_argument("--family", type=str, default="ffg", choices=["ffg"])
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=50)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()

run_name = f"vae_{args.family}_h{'-'.join(map(str, args.hidden_dims))}_{args.act}_seed{args.seed}"
config = vars(args) | {"model": "vae"}          # tous les arguments vont dans wandb


torch.manual_seed(args.seed)

data = torch.load("/kaggle/working/mnist_bin.pt")
X_train, X_val = data['train'].to(device), data['val'].to(device)


N = X_train.shape[0] #amount of training data 
B = args.batch_size  #batch size (Cremer et al = 50)

input_dim = X_train.shape[1]
hidden_dims = args.hidden_dims
latent_dim = args.latent_dim
output_dim = input_dim

act = args.act
lr = args.lr

enc, dec = encoder(input_dim, hidden_dims, latent_dim, act).to(device), decoder(latent_dim, hidden_dims, output_dim, act).to(device)
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr = args.lr)
epochs = args.epochs

family = args.family

run = wandb.init(project=args.project, config=config, name=run_name)

best_val = float("inf")  
for epoch in range(epochs):

    train_loss_tot, val_loss_tot = 0, 0
    perm = torch.randperm(N, device = device) #suffle the training data order at each epoch
    
    #training loop
    for i in range(0, N, B): #separation with batches
        x_batch = X_train[perm[i:i+B]]
        train_loss = - ELBO(x_batch, enc, dec) #mean on the batch
        train_loss_tot += train_loss.item()*x_batch.shape[0] #sum of the train loss on the batch
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    #validation loop
    with torch.no_grad():
        val_loss_tot = - ELBO(X_val, enc, dec).item() #average validation loss on the whole validation set
    
    if val_loss_tot < best_val:
        best_val = val_loss_tot
        torch.save(enc.state_dict(), f"/kaggle/working/{run_name}_enc_vae.pt")
        torch.save(dec.state_dict(), f"/kaggle/working/{run_name}_dec_vae.pt")

    wandb.log({"epoch": epoch, "train_loss": train_loss_tot/N, "val_loss": val_loss_tot})
 
art = wandb.Artifact(name=run_name, type="model",
                     metadata={"best_validation": best_val})
art.add_file(f"/kaggle/working/{run_name}_enc_vae.pt")
art.add_file(f"/kaggle/working/{run_name}_dec_vae.pt")
run.log_artifact(art)

#save best weights of the model on wandB directly

run.finish() # wandb end




