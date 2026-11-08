import wandb
run = wandb.init(entity="bastienjacques38-university-", project="PGM")
artifact = run.use_artifact("bastienjacques38-university-/PGM/vae_ffg_h100_tanh_seed0:v0", type="model")
artifact_dir = artifact.download()