# MVA3842: Introduction to Probabilistic Graphical Models and Deep Generative Models

Course materials, notes, and exercises for probabilistic graphical models and deep generative models.

Project: amortization and inference gaps in VAEs, based on

- Cremer, Li & Duvenaud (2018), *Inference Suboptimality in Variational Autoencoders*
- Margossian & Blei (2024), *Amortized Variational Inference: When and Why?*

## Setup

This repo has its own environment, managed with [uv](https://docs.astral.sh/uv/) (Python 3.12, pinned in `.python-version`). Dependencies are listed in `pyproject.toml`.

```bash
uv sync                                   # create .venv and install everything
source .venv/bin/activate                 # optional: otherwise prefix commands with `uv run`
```

Secrets go in `.env`, which git ignores and the code loads with `python-dotenv`:

```bash
echo "WANDB_API_KEY=<your-key>" > .env    # or: cp ../.env .env
```

Jupyter kernel (for VS Code / JupyterLab):

```bash
uv run python -m ipykernel install --user --name mva3842 --display-name "MVA3842 (VAE)"
```

## Everyday commands

```bash
uv run python script.py                   # run a script inside the env
uv run jupyter lab                        # start JupyterLab
uv add <package>                          # add a dependency (updates pyproject.toml + uv.lock)
uv run pytest                             # tests (pytest and ruff are in the dev group, installed by default)
```
