import torch
import numpy as np
import matplotlib.pyplot as plt

from models.mmppp import IMMPpp
from models.modules import FC_vec

from loader.Toy_dataset import Toy, toy_visualizer, pallete
from models.lbf import LfD, Gaussian_basis, phi, vbf

device = 'cuda:0'

ds = Toy(root='datasets/EXP2')
dl = torch.utils.data.DataLoader(ds, batch_size=5)
fig, axs = plt.subplots(1, 1, figsize=(12, 3))
toy_visualizer(ds.env, axs, traj=ds.data, label=ds.targets)
plt.show()

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

mmppp = IMMPpp(
    encoder,
    decoder,
    dof=2,
    b=30,
    h_mul=1,
    iso_reg=1,
    basis='Gaussian',
    mode='vmp',
    via_points=[[0.8, 0.8], [-0.8, -0.8]]
)
mmppp.to(device)

opt = torch.optim.Adam(mmppp.parameters(), lr=0.0001)
for epoch in range(1000):
    for x, y in dl:
        train_results = mmppp.train_step(x.to(device), optimizer=opt)
        train_loss = train_results["loss"]
    if epoch%50 == 0:
        print(f"[Epoch: {epoch}] Loss: {train_loss}")

n_components = 3
n_samples = 300
alpha1 = 0.5
alpha2 = 0.3

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

toy_visualizer(
    ds.env, 
    axs[0], 
    traj=ds.data, 
    label=ds.targets, 
    alpha=alpha1)

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
plt.show()

n_components = 3
n_samples = 300
alpha1 = 0.5
alpha2 = 0.3

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

toy_visualizer(
    ds.env, 
    axs[0], 
    traj=ds.data, 
    label=ds.targets, 
    alpha=alpha1)

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

mmppp.fit_GMM(ds.data.to(device), n_components=n_components)
mmppp.gmm_thr = mmppp.gmm_thr + 1.5 ## to exclude samples from distrib. tails 
dict_samples = mmppp.sample(
    n_samples, 
    device=device, 
    traj_len=201,
    clipping=True
)
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

    zz = np.exp(np.diagonal(
        -(delta@np.linalg.inv(cov)@delta.transpose())
    )/2)
    
    c = sample_y[sample_y == i].reshape(-1,1)
    contour = axs[1].contour(
        xx.reshape(40, 40), 
        yy.reshape(40, 40), 
        zz.reshape(40, 40),
        levels=[0.3, 0.5, 0.7, 0.9, 0.9999], 
        colors=pallete[10-i],
        linewidths=2,
        )
    
toy_visualizer(
    ds.env, 
    axs[2], 
    traj=q_traj, 
    label=10-dict_samples['cluster_samples'], 
    alpha=alpha2)
plt.show()
