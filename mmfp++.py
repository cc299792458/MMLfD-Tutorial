import torch
import numpy as np
import matplotlib.pyplot as plt

from models.mmppp import MMPpp
from models.modules import FC_vec
from flow_matching.utils import ModelWrapper
from flow_matching.solver import Solver, ODESolver
from models.lbf import LfD, Gaussian_basis, phi, vbf
from torch.utils.data import TensorDataset, DataLoader
from loader.Toy_dataset import Toy, toy_visualizer, pallete
from conditional_flow_matching import ConditionalFlowMatching

# Smooth loss curve using moving average
def smooth_loss(loss_vals, window=100):
    return np.convolve(loss_vals, np.ones(window)/window, mode='valid')

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None, device='cpu', **extras):
        batch_size = x.size(0)
        t = t.reshape(-1, 1).expand(batch_size, 1)
        
        return self.model.velocity(x, t, cond, device=device)

# Device configuration
device = 'cuda:0'
num_dof = 2
latent_dim = 2
num_basis = 30
traj_len = 201

# Initialize dataset
ds = Toy(root='datasets/EXP2')
training_data = ds.data[:12]
training_targets = ds.targets[:12]
eval_data = ds.data[12:]
eval_targets = torch.tensor([10, 10, 10])

# Initial visualization of dataset
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
toy_visualizer(ds.env, axs, traj=ds.data, label=torch.concat([training_targets, eval_targets]))
axs.set_title("Demonstration trajectories")
plt.show()

# Basis function calculations
tau = torch.linspace(0, 1, 100).view(1, -1, 1)
basis_values = Gaussian_basis(tau)
phi_values = phi(basis_values)

# LfD initialization
training_w = LfD(training_data, mode='vmp', via_points=[[0.8, 0.8], [-0.8, -0.8]])
eval_w = LfD(eval_data, mode='vmp', via_points=[[0.8, 0.8], [-0.8, -0.8]])

# VMP trajectory reconstruction
vmp_recon_curve = vbf(tau, phi_values, torch.concat([training_w, eval_w], dim=0), via_points=[[0.8, 0.8], [-0.8, -0.8]])

# DataLoader setup
dl = torch.utils.data.DataLoader(ds, batch_size=5)

# Plot: VMP reconstruction
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
toy_visualizer(ds.env, axs[0], traj=ds.data, label=torch.concat([training_targets, eval_targets]))
toy_visualizer(ds.env, axs[1], traj=ds.data, label=torch.tensor(len(ds.targets)*[10]), alpha=0.5)
toy_visualizer(ds.env, axs[1], traj=vmp_recon_curve, label=torch.concat([training_targets, eval_targets]))
axs[0].set_title("Demonstration trajectories")
axs[1].set_title("VMP recon trajectories")
plt.show()

# Model components
encoder = FC_vec(
    in_chan=num_dof * num_basis, out_chan=latent_dim, l_hidden=[512, 512], activation=['elu', 'elu'], out_activation='linear'
)
decoder = FC_vec(
    in_chan=latent_dim, out_chan=num_dof * num_basis, l_hidden=[512, 512], activation=['elu', 'elu'], out_activation='linear'
)

# MMPpp model initialization
mmppp = MMPpp(
    encoder, decoder, dof=num_dof, b=num_basis, h_mul=1, basis='Gaussian', mode='vmp', via_points=[[0.8, 0.8], [-0.8, -0.8]]
).to(device)

# Training loop
opt = torch.optim.Adam(mmppp.parameters(), lr=0.0001)
for epoch in range(1000):
    for x, y in dl:
        training_results = mmppp.train_step(x.to(device), optimizer=opt)
        training_loss = training_results["loss"]
        eval_loss = mmppp.loss_func(eval_w.view(eval_w.size(0), -1).to(device)).item()

    if epoch % 100 == 0:
        print(f"[Epoch: {epoch}] Training Loss: {training_loss:.5f}| Eval Loss: {eval_loss:.5f}")

# Plot: AE reconstruction
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(ds.env, axs[0], traj=ds.data, label=ds.targets)
axs[0].set_title("Demonstration trajectories")

# Subplot 2: Latent space (gray points)
w = mmppp.get_w_from_traj(ds.data.to(device))
latent_values = mmppp.encode(w).detach().cpu()
color = [pallete[data] for data in torch.concat([training_targets, eval_targets]).numpy()]
axs[1].scatter(latent_values[:,0], latent_values[:,1], c=color, marker='x')
axs[1].axis('equal')
axs[1].axis('off')
axs[1].set_title("Latent space")

# Subplot 3: AE reconstructed trajectories
recon_w = mmppp.decode(mmppp.encode(mmppp.get_w_from_traj(ds.data.to(device)))).detach().cpu()
ae_recon_curve = vbf(tau, phi_values, recon_w.view(len(recon_w), -1, 2), via_points=[[0.8, 0.8], [-0.8, -0.8]])

toy_visualizer(ds.env, axs[2], traj=ds.data, label=torch.tensor(len(ds.targets)*[10]), alpha=0.5)
toy_visualizer(ds.env, axs[2], traj=ae_recon_curve, label=torch.concat([training_targets, eval_targets]))
axs[2].set_title("AE recon trajectories")
plt.show()

n_components = 3
n_samples = 100

# Plot: GMM reconstruction
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(ds.env, axs[0], traj=ds.data, label=torch.concat([training_targets, eval_targets]))
axs[0].set_title("Demonstration trajectories")

# Subplot 2: Latent space with samples
axs[1].scatter(latent_values[:,0], latent_values[:,1], c=color, marker='x')
axs[1].axis('equal')
axs[1].axis('off')

mmppp.fit_GMM(ds.data.to(device), n_components=n_components)
dict_samples = mmppp.sample(n_samples, device=device, traj_len=traj_len)
q_traj = dict_samples['q_traj_samples'].detach().cpu()
z_samples = dict_samples['z_samples'].detach().cpu()
sample_y = dict_samples['cluster_samples'].numpy()

# Additional latent points with colors
axs[1].scatter(
    z_samples[:,0], z_samples[:,1],
    c=[pallete[i] for i in 10-dict_samples['cluster_samples']],
    alpha=0.6, marker='*',
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
    
# Subplot 3: Generated trajectories
toy_visualizer(ds.env, axs[2], traj=q_traj, label=10-dict_samples['cluster_samples'],)
axs[2].set_title("Generated trajectories")
plt.show()

# Prepare data
z = encoder(w).detach()
z_mean, z_std = z.mean(dim=0), z.std(dim=0)
z = (z - z_mean) / (z_std + 1e-8)
z_dataset = TensorDataset(z, ds.targets)
z_dataloader = DataLoader(z_dataset, batch_size=15)

epochs = 5_000

cfm = ConditionalFlowMatching(input_dim=latent_dim, cond_dim=0).to(device)
opt = torch.optim.Adam(cfm.parameters(), lr=1e-4, weight_decay=1e-6)

loss_history = []

for epoch in range(epochs):
    for x_1, label in z_dataloader:
        opt.zero_grad()
        loss, x_t, dx_t_pred, dx_t = cfm.compute_loss(x1=x_1)
        
        # Record the loss value (use .item() to get a scalar)
        loss_history.append(loss.item())
        
        # Backpropagation & optimization
        loss.backward()
        opt.step()

    # Optional: Print average loss per epoch
    if epoch % 100 == 0:  # Print every 100 epochs
        avg_loss = sum(loss_history[-len(z_dataloader):]) / len(z_dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

window_size = 100

# Plot Unconditional CFM Loss
plt.figure(figsize=(10,5))
smoothed = smooth_loss(loss_history, window_size)
plt.plot(loss_history, alpha=0.2, label="Raw")  # Faint original
plt.plot(smoothed, label="Smoothed")  # Main smoothed curve
plt.title("Unconditional CFM Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Flow-Based Model reconstruction
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(ds.env, axs[0], traj=ds.data, label=torch.concat([training_targets, eval_targets]))
axs[0].set_title("Demonstration trajectories")

wrapped_cfm = WrappedModel(cfm)
solver = ODESolver(velocity_model=wrapped_cfm)

x_init = torch.randn((n_samples, latent_dim), dtype=torch.float32, device=device)
time_grid = torch.linspace(0, 1, 10)
sol = solver.sample(x_init=x_init, time_grid=time_grid, method='midpoint', step_size=0.01, device=device)

# z_samples = cfm.sample(n_samples, device=device) * z_std + z_mean
z_samples = sol * z_std + z_mean
recon_w = decoder(z_samples).cpu()
q_traj = vbf(tau, phi_values, recon_w.view(len(recon_w), -1, 2), via_points=[[0.8, 0.8], [-0.8, -0.8]])

# Subplot 2: Latent space with samples
axs[1].scatter(latent_values[:,0], latent_values[:,1], c=color, marker='x', s=100)
axs[1].axis('equal')
axs[1].axis('off')

# Additional latent points
axs[1].scatter(z_samples.cpu()[:,0], z_samples.cpu()[:,1], alpha=0.6, marker='*', color='grey')
axs[1].set_title("Latent space")
# Subplot 3: Generated trajectories
toy_visualizer(ds.env, axs[2], traj=q_traj.detach(), label=torch.ones(q_traj.size(0), dtype=torch.int) * 10, alpha=0.5)
axs[2].set_title("Generated trajectories")
plt.show()

cfm = ConditionalFlowMatching(input_dim=latent_dim, cond_dim=1).to(device)
opt = torch.optim.Adam(cfm.parameters(), lr=1e-4, weight_decay=1e-6)

loss_history = []

for epoch in range(epochs):
    for x_1, label in z_dataloader:
        opt.zero_grad()
        label = label.reshape(-1, 1).to(device)
        loss, x_t, dx_t_pred, dx_t = cfm.compute_loss(x1=x_1, cond=label)
        
        # Record the loss value (use .item() to get a scalar)
        loss_history.append(loss.item())
        
        # Backpropagation & optimization
        loss.backward()
        opt.step()

    # Optional: Print average loss per epoch
    if epoch % 100 == 0:  # Print every 100 epochs
        avg_loss = sum(loss_history[-len(z_dataloader):]) / len(z_dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

# Plot the loss curve after training
plt.figure(figsize=(10, 5))
smoothed = smooth_loss(loss_history, window_size)
plt.plot(loss_history, alpha=0.2, label="Raw")  # Faint original
plt.plot(smoothed, label="Smoothed")  # Main smoothed curve
plt.title("Conditional CFM Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Flow-Based Model reconstruction
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Original trajectories
toy_visualizer(ds.env, axs[0], traj=ds.data, label=torch.concat([training_targets, eval_targets]))
axs[0].set_title("Demonstration trajectories")

wrapped_cfm = WrappedModel(cfm)
solver = ODESolver(velocity_model=wrapped_cfm)

x_init = torch.randn((n_samples, latent_dim), dtype=torch.float32, device=device)
time_grid = torch.linspace(0, 1, 10)
cond_samples = torch.arange(100, device=device) % 3
sol = solver.sample(x_init=x_init, time_grid=time_grid, method='midpoint', step_size=0.01, cond=cond_samples.reshape(-1, 1), device=device)

# z_samples = cfm.sample(n_samples, cond_samples.reshape(-1, 1), device=device) * z_std + z_mean
z_samples = sol * z_std + z_mean
recon_w = decoder(z_samples).cpu()
q_traj = vbf(tau, phi_values, recon_w.view(len(recon_w), -1, 2), via_points=[[0.8, 0.8], [-0.8, -0.8]])

# Subplot 2: Latent space with samples
axs[1].scatter(latent_values[:,0], latent_values[:,1], c=color, marker='x', s=100)
axs[1].axis('equal')
axs[1].axis('off')

# Additional latent points
axs[1].scatter(z_samples.cpu()[:,0], z_samples.cpu()[:,1], alpha=0.6, marker='*', color=[pallete[i] for i in cond_samples])
axs[1].set_title("Latent space")
# Subplot 3: Generated trajectories
toy_visualizer(ds.env, axs[2], traj=q_traj.detach(), label=cond_samples, alpha=0.5)
axs[2].set_title("Generated trajectories")
plt.show()