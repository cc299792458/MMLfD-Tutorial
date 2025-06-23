import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mmppp import MMPpp
from models.modules import FC_vec
from loader.Toy_dataset import Toy, toy_visualizer, pallete
from models.lbf import LfD, Gaussian_basis, phi, vbf

# Device configuration
device = 'cuda:0'

# Initialize dataset
ds = Toy(root='datasets/EXP2')

# Initial visualization of dataset
fig, axs = plt.subplots(1, 1, figsize=(12, 3))
toy_visualizer(ds.env, axs, traj=ds.data, label=ds.targets)
axs.set_title("demonstration trajectories")
plt.show()

# LfD initialization
w = LfD(ds.data, mode='vmp', via_points=[[0.8, 0.8], [-0.8, -0.8]])

# Basis function calculations
z = torch.linspace(0, 1, 100).view(1, -1, 1)
basis_values = Gaussian_basis(z)
phi_values = phi(basis_values)

# VMP trajectory reconstruction
vmp_recon_curve = vbf(z, phi_values, w, via_points=[[0.8, 0.8], [-0.8, -0.8]])

# DataLoader setup
dl = torch.utils.data.DataLoader(ds, batch_size=5)

# Comparison of original and reconstructed trajectories
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
toy_visualizer(ds.env, axs[0], traj=ds.data, label=ds.targets)
toy_visualizer(ds.env, axs[1], traj=vmp_recon_curve, label=ds.targets)
axs[0].set_title("demonstration trajectories")
axs[1].set_title("VMP recon trajectories")
plt.show()

# Model components
encoder = FC_vec(
    in_chan=60,
    out_chan=2,
    l_hidden=[512, 512],
    activation=['elu', 'elu'],
    out_activation='linear'
)
decoder = FC_vec(
    in_chan=2,
    out_chan=60,
    l_hidden=[512, 512],
    activation=['elu', 'elu'],
    out_activation='linear'
)

# MMPpp model initialization
mmppp = MMPpp(
    encoder,
    decoder,
    dof=2,
    b=30,
    h_mul=1,
    basis='Gaussian',
    mode='vmp',
    via_points=[[0.8, 0.8], [-0.8, -0.8]]
)
mmppp.to(device)

# Training loop
opt = torch.optim.Adam(mmppp.parameters(), lr=0.0001)
for epoch in range(1000):
    for x, y in dl:
        train_results = mmppp.train_step(x.to(device), optimizer=opt)
        train_loss = train_results["loss"]
    if epoch%100 == 0:
        print(f"[Epoch: {epoch}] Loss: {train_loss}")

# =============================================================================
# Visualization Section 1
# =============================================================================
n_components = 3
n_samples = 300
alpha1 = 0.5
alpha2 = 0.3

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(ds.env, axs[0], traj=ds.data, label=ds.targets, alpha=alpha1)
axs[0].set_title("demonstration trajectories")

# Subplot 2: Latent space (gray points)
w = mmppp.get_w_from_traj(ds.data.to(device))
latent_values = mmppp.encode(w).detach().cpu()
color = [pallete[data] for data in ds.targets.cpu().numpy()]
axs[1].scatter(latent_values[:,0], latent_values[:,1], c=color, marker='x')
axs[1].axis('equal')
axs[1].axis('off')
axs[1].set_title("Latent space")

# Subplot 3: VMP reconstructed trajectories
mmppp.fit_GMM(ds.data.to(device), n_components=n_components)
dict_samples = mmppp.sample(n_samples, device=device, traj_len=201)
q_traj = dict_samples['q_traj_samples'].detach().cpu()
z_samples = dict_samples['z_samples'].detach().cpu()
sample_y = dict_samples['cluster_samples'].numpy()

recon_w = mmppp.decode(mmppp.encode(mmppp.get_w_from_traj(ds.data.to(device)))).detach().cpu()
basis_values = Gaussian_basis(z)
phi_values = phi(basis_values)
vmp_recon_curve = vbf(z, phi_values, recon_w.view(len(recon_w), -1, 2), via_points=[[0.8, 0.8], [-0.8, -0.8]])

toy_visualizer(
    ds.env, 
    axs[2], 
    traj=vmp_recon_curve, 
    label=torch.tensor(len(ds.targets)*[10]), 
    alpha=alpha1)
axs[2].set_title("VMP recon trajectories")
plt.show()

# =============================================================================
# Visualization Section 2
# =============================================================================
n_components = 3
n_samples = 300
alpha1 = 0.5
alpha2 = 0.3

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(ds.env, axs[0], traj=ds.data, label=ds.targets, alpha=alpha1)
axs[0].set_title("demonstration trajectories")

# Subplot 2: Latent space with samples
w = mmppp.get_w_from_traj(ds.data.to(device))
latent_values = mmppp.encode(w).detach().cpu()
axs[1].scatter(latent_values[:,0], latent_values[:,1], c=color, marker='x')
axs[1].axis('equal')
axs[1].axis('off')

mmppp.fit_GMM(ds.data.to(device), n_components=n_components)
dict_samples = mmppp.sample(n_samples, device=device, traj_len=201)
q_traj = dict_samples['q_traj_samples'].detach().cpu()
z_samples = dict_samples['z_samples'].detach().cpu()
sample_y = dict_samples['cluster_samples'].numpy()

# Additional latent points with colors
axs[1].scatter(
    z_samples[:,0], 
    z_samples[:,1],
    c=[pallete[i] for i in 10-dict_samples['cluster_samples']],
    alpha=0.6,
    marker='*',
)
axs[1].set_title("Latent space")

# GMM contour plots
xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
for i, (mu, cov) in enumerate(zip(mmppp.gmm.means_, mmppp.gmm.covariances_)):
    tempZ = z_samples[sample_y == i]
    x = np.linspace(xmin, xmax, 40)
    y = np.linspace(ymin, ymax, 40)
    xx, yy = np.meshgrid(x, y) 
    delta = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1) - mu.reshape(1, -1)
    
    zz = np.exp(np.diagonal(-(delta@np.linalg.inv(cov)@delta.transpose())/2))
    
    c = sample_y[sample_y == i].reshape(-1,1)
    contour = axs[1].contour(
        xx.reshape(40, 40), 
        yy.reshape(40, 40), 
        zz.reshape(40, 40),
        levels=[0.3, 0.5, 0.7, 0.9, 0.9999], 
        colors=pallete[10-i],
        linewidths=2,
    )
    
# Subplot 3: Empty trajectories
recon_w = mmppp.decode(mmppp.encode(mmppp.get_w_from_traj(ds.data.to(device)))).detach().cpu()
basis_values = Gaussian_basis(z)
phi_values = phi(basis_values)
vmp_recon_curve = vbf(z, phi_values, recon_w.view(len(recon_w), -1, 2), via_points=[[0.8, 0.8], [-0.8, -0.8]])

toy_visualizer(
    ds.env, 
    axs[2], 
    traj=torch.tensor([]), 
    label=torch.tensor([]), 
    alpha=alpha1)
axs[2].set_title("Empty trajectories")
plt.show()

# =============================================================================
# Visualization Section 3
# =============================================================================
n_components = 3
n_samples = 300
alpha1 = 0.5
alpha2 = 0.3

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(
    ds.env, 
    axs[0], 
    traj=ds.data, 
    label=ds.targets, 
    alpha=alpha1)
axs[0].set_title("demonstration trajectories")

# Subplot 2: Colored latent space
w = mmppp.get_w_from_traj(ds.data.to(device))
latent_values = mmppp.encode(w).detach().cpu()
axs[1].scatter(
    latent_values[:,0], 
    latent_values[:,1],
    c=[pallete[i] for i in ds.targets],
    marker='x',
)
axs[1].axis('equal')
axs[1].axis('off')
axs[1].set_title("Latent space")

# Sampling and GMM contours
mmppp.fit_GMM(ds.data.to(device), n_components=n_components)
dict_samples = mmppp.sample(n_samples, device=device, traj_len=201)
q_traj = dict_samples['q_traj_samples'].detach().cpu()
z_samples = dict_samples['z_samples'].detach().cpu()

axs[1].scatter(
    z_samples[:,0], 
    z_samples[:,1],
    c=[pallete[i] for i in 10-dict_samples['cluster_samples']],
    alpha=0.6,
    marker='*',
)

sample_y = dict_samples['cluster_samples'].numpy()
xmin, xmax = axs[1].get_xbound()
ymin, ymax = axs[1].get_ybound()
for i, (mu, cov) in enumerate(zip(mmppp.gmm.means_, mmppp.gmm.covariances_)):
    tempZ = z_samples[sample_y == i]
    x = np.linspace(xmin, xmax, 40)
    y = np.linspace(ymin, ymax, 40)
    xx, yy = np.meshgrid(x, y) 
    delta = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1) - mu.reshape(1, -1)
    
    zz = np.exp(np.diagonal(-(delta@np.linalg.inv(cov)@delta.transpose())/2))
    
    c = sample_y[sample_y == i].reshape(-1,1)
    contour = axs[1].contour(
        xx.reshape(40, 40), 
        yy.reshape(40, 40), 
        zz.reshape(40, 40),
        levels=[0.3, 0.5, 0.7, 0.9, 0.9999], 
        colors=pallete[10-i],
        linewidths=2,
    )
    
# Subplot 3: Generated trajectories
toy_visualizer(
    ds.env, 
    axs[2], 
    traj=q_traj, 
    label=10-dict_samples['cluster_samples'], 
    alpha=alpha2)
axs[2].set_title("Generated trajectories")
plt.show()

# =============================================================================
# Visualization Section 4 (with commented content)
# =============================================================================
n_components = 3
n_samples = 300
alpha1 = 0.5
alpha2 = 0.3

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Original trajectories
toy_visualizer(
    ds.env, 
    axs[0], 
    traj=ds.data, 
    label=ds.targets, 
    alpha=alpha1)
axs[0].set_title("demonstration trajectories")

# Subplot 2: Colored latent space
w = mmppp.get_w_from_traj(ds.data.to(device))
latent_values = mmppp.encode(w).detach().cpu()
axs[1].scatter(
    latent_values[:,0], 
    latent_values[:,1],
    c=[pallete[i] for i in ds.targets],
    marker='x',
)
axs[1].axis('equal')
axs[1].axis('off')

# Sampling (commented out section below)
mmppp.fit_GMM(ds.data.to(device), n_components=n_components)
dict_samples = mmppp.sample(n_samples, device=device, traj_len=201)
q_traj = dict_samples['q_traj_samples'].detach().cpu()
z_samples = dict_samples['z_samples'].detach().cpu()
axs[1].set_title("Latent space")
plt.show()

# #############################################################################
# # This section is intentionally kept commented as in the original code

# axs[1].scatter(
#     z_samples[:,0], 
#     z_samples[:,1],
#     c=[pallete[i] for i in 10-dict_samples['cluster_samples']],
#     alpha=0.6,
#     marker='*',
#     )

# sample_y = dict_samples['cluster_samples'].numpy()

# xmin, xmax = axs[1].get_xbound()
# ymin, ymax = axs[1].get_ybound()
# for i, (mu, cov) in enumerate(zip(mmppp.gmm.means_, mmppp.gmm.covariances_)):
#     tempZ = z_samples[sample_y == i]
#     x = np.linspace(xmin, xmax, 40)
#     y = np.linspace(ymin, ymax, 40)
#     xx, yy = np.meshgrid(x, y) 
#     delta = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1) - mu.reshape(1, -1)

#     zz = np.exp(np.diagonal(
#         -(delta@np.linalg.inv(cov)@delta.transpose())
#     )/2)
    
#     c = sample_y[sample_y == i].reshape(-1,1)
#     contour = axs[1].contour(
#         xx.reshape(40, 40), 
#         yy.reshape(40, 40), 
#         zz.reshape(40, 40),
#         levels=[0.3, 0.5, 0.7, 0.9, 0.9999], 
#         colors=pallete[10-i],
#         linewidths=2,
#         )
    
# toy_visualizer(
#     ds.env, 
#     axs[2], 
#     traj=q_traj, 
#     label=10-dict_samples['cluster_samples'], 
#     alpha=alpha2)
#############################################################################