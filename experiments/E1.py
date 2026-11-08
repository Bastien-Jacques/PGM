import numpy as np
import matplotlib.pyplot as plt

## Modèle professeur

# (z1, z2) ~ N(0,I) et x1 | z ~ N(z1^2, Sigma1^2)  x2 | z ~ N(z1, sigma2^2)  x3 | z ~ N(z1 + z2, sigma3^2) 
SIGMA = np.array([0.5, 2.0, 0.3])          # écarts-types du bruit sur x1, x2, x3

def decoder_mean(z1, z2): #moyenne du decodeur pour x1, x2, x3
    #Mean of p(x | z) : g(z) = (z1^2, z1, z1 + z2)
    return np.stack([z1**2, z1, z1 + z2], axis=-1)


def log_prior(z1, z2):  # log p(z) = log p(z1) + log p(z2) avec z ~ N(0,I)
    return -0.5 * (z1**2 + z2**2) - np.log(2 * np.pi)

def log_lik(x, z1, z2): #log likelihood obtenue log p(x|z) 
    #log p(x | z) pour un x de dimension 3, évalué en tous les z de la grille
    m = decoder_mean(z1, z2)
    return np.sum(-0.5 * ((x - m) / SIGMA) ** 2 - np.log(SIGMA * np.sqrt(2 * np.pi)), axis=-1)


## grille en z 
L, n = 4.0, 401
g = np.linspace(-L, L, n)
Z1, Z2 = np.meshgrid(g, g, indexing="ij")
dA = (g[1] - g[0]) ** 2                     # aire d'une cellule

def posterior_on_grid(x): #return p(z|x) en utilisant Bayes: log p(z|x) = log p(z) + log p(x|z) - log p(x). The 'hard' part is to compute log p(x)

    # we compute p(xi) on the grid p(xi) = int p(z) p(x|z) dz ~= Sum_{k \in grid} p(z_k)p(x_i|z_k)dA
    """Renvoie p(z | x) sur la grille et log p(x) (par quadrature)."""
    log_joint = log_prior(Z1, Z2) + log_lik(x, Z1, Z2)
    m = log_joint.max()                      # pour la stabilité numérique
    log_px = m + np.log(np.sum(np.exp(log_joint - m)) * dA)
    return np.exp(log_joint - log_px), log_px

## figure
xs = [np.array([0.2, 0.0, 0.5]),            # x1 petit : unimodal
      np.array([2.5, 0.0, 0.0]),            # x1 grand, x2 neutre : deux modes équilibrés
      np.array([2.5, 1.5, 0.0]),            # x2 > 0 : mode z1 > 0 favorisé
      np.array([2.5, 0.0, 2.0])]            # x3 décale le long de z1 + z2

def plot_posterior(ax, x, title=None):
    """Vrai posterior en contours verts pleins, comme la Figure 2 de Cremer."""
    post, log_px = posterior_on_grid(x)
    levels = post.max() * np.array([0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.contourf(Z1, Z2, post, levels=levels, cmap="Greens")          # remplissage vert
    ax.contour(Z1, Z2, post, levels=levels[:-1], colors="darkgreen", linewidths=0.5)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])                             # comme chez Cremer
    if title is not None:
        ax.set_title(title, fontsize=9)
    return post, log_px

def plot_approx(ax, q_density): #A faire avec l'encodeur
    """Approximation q (évaluée sur la même grille) en contours bleus."""
    levels = q_density.max() * np.array([0.05, 0.2, 0.5, 0.8])
    ax.contour(Z1, Z2, q_density, levels=levels, colors="blue", linewidths=1.0)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))

for ax, x in zip(axes, xs):
    post, log_px = plot_posterior(ax, x, title=f"x = {np.round(x, 2)}")
    print(f"x = {x},  log p(x) = {log_px:.3f},  masse totale = {post.sum() * dA:.4f}")
axes[0].set_ylabel("True posterior")
plt.tight_layout(); plt.show()

