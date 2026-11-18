import torch
from model import Encoder, Decoder
from objectives import elbo
import wandb 
import argparse
from utils import load_dataset, load_vae

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#parameters

p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default = "mnist_bin")
p.add_argument("--hidden_dim_enc", type=int, nargs="+", default=[100]) 
p.add_argument("--hidden_dim_dec", type=int, nargs="+", default=[100]) 
p.add_argument("--latent_dim", type=int, default=2)
p.add_argument("--act", type=str, default="tanh", choices=["tanh", "relu", "elu"])
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch_size", type=int, default=50)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--family", type = str, default = "ffg")
p.add_argument("--project", type=str, default="PGM")
args = p.parse_args()


name_vae = "vae_ffg_h100_tanh_seed0" #run with the Cremer et al decoder

run_name = f"enc_{args.data}_{args.family}_h{'-'.join(map(str, args.enc_hidden_dims))}_{args.act}_dec-{name_vae}_seed{args.seed}"
config = vars(args) | {"model": "vae"}          # tous les arguments vont dans wandb


torch.manual_seed(args.seed)

data = load_dataset(args.data)
if args.data == "mnist_bin":
    X, X_val = data["train"].to(device), data["val"].to(device)
elif args.data.startswith("morph"):
    X, X_val = data["X"].to(device), None
N = X.shape[0]

B = args.batch_size  #batch size (Cremer et al = 50)

input_dim = X.shape[1]
hidden_dim_enc = args.hidden_dim_enc
hidden_dim_dec = args.hidden_dim_dec
latent_dim = args.latent_dim
output_dim = input_dim

act = args.act
lr = args.lr

enc, dec = load_vae(name_vae,hidden_dim_enc = hidden_dim_enc, hidden_dim_dec = hidden_dim_dec, latent_dim=2, act="tanh", input_dim=784)
dec = dec.to(device).requires_grad_(False) #gradient not needed since the decoder is already trained: frozen decoder

if args.data == "mnist_bin":
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr = args.lr)
else:
    optimizer = torch.optim.Adam(list(enc.parameters()), lr = args.lr) #for the encoder training only 


epochs = args.epochs

family = args.family

run = wandb.init(project=args.project, config=config, name=run_name)

best_val = float("inf")  
for epoch in range(epochs):

    train_loss_tot, val_loss_tot = 0, 0
    perm = torch.randperm(N, device = device) #suffle the training data order at each epoch
    
    #training loop
    for i in range(0, N, B): #separation with batches
        x_batch = X[perm[i:i+B]]
        train_loss = - elbo(x_batch, enc, dec) #mean on the batch
        train_loss_tot += train_loss.item()*x_batch.shape[0] #sum of the train loss on the batch
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    #validation loop
    if args.data == 'mnist_bin':
        with torch.no_grad():
            val_loss_tot = - elbo(X_val, enc, dec).item() #average validation loss on the whole validation set
        
        if val_loss_tot < best_val:
            best_val = val_loss_tot
            torch.save(enc.state_dict(), f"/kaggle/working/{run_name}_enc_vae.pt")
            torch.save(dec.state_dict(), f"/kaggle/working/{run_name}_dec_vae.pt")

        wandb.log({"epoch": epoch, "train_loss": train_loss_tot/N, "val_loss": val_loss_tot})
    wandb.log({"epoch": epoch, "loss": train_loss_tot/N}) #wandb log in the case of morphing dataset
 
art = wandb.Artifact(name=run_name, type="model",
                     metadata={"best_validation": best_val})
art.add_file(f"/kaggle/working/{run_name}_enc_vae.pt")
art.add_file(f"/kaggle/working/{run_name}_dec_vae.pt")
run.log_artifact(art)

#save best weights of the model on wandB directly

run.finish() # wandb end







