import torch
import copy
import matplotlib.pyplot as plt
from loader.Pouring_dataset import Pouring
from loader.Toy_dataset import Toy, toy_visualizer
from models.mmp import NRMMP
from models.modules import FC_SE32vec, FC_vec2SE3
from models.modules import FC_traj2latent, FC_latent2traj

# Configuration
device = 'cuda:0'
use_pretrained = False
pouring = True  # True means Pouring, False means Toy

def main():
    # Initialize dataset and model
    if pouring:
        ds = Pouring(graph={'bs_nn': 2, 'num_nn': 6, 'include_center': True, 'replace': False})

        encoder = FC_SE32vec(
            in_chan=480 * 12,
            out_chan=2,
            l_hidden=[2048, 1024, 512, 256],
            activation=['gelu', 'gelu', 'gelu', 'gelu'],
            out_activation='linear'
        )
        
        decoder = FC_vec2SE3(
            in_chan=2,
            out_chan=480 * 6,
            l_hidden=[256, 512, 1024, 2048],
            activation=['gelu', 'gelu', 'gelu', 'gelu'],
            out_activation='linear'
        )
    
    else:
        ds = Toy()
        encoder = FC_traj2latent(
            in_chan=201 * 2,
            out_chan=2,
            l_hidden=[2048, 1024, 512, 256],
            activation=['gelu', 'gelu', 'gelu', 'gelu'],
            out_activation='linear'
        )
        decoder = FC_latent2traj(
            in_chan=2,
            out_chan=201 * 2,
            l_hidden=[256, 512, 1024, 2048],
            activation=['gelu', 'gelu', 'gelu', 'gelu'],
            out_activation='linear'
        )
    
    dl = torch.utils.data.DataLoader(ds, batch_size=10)
    
    mmp = NRMMP(encoder, decoder, approx_order=1, type_='SE3' if pouring else 'linear', kernel={'type': 'binary', 'lambda':0.05})
    mmp.to(device)

    # Training or loading pretrained model
    if use_pretrained:
        load_dict = torch.load("results/nrmmp.pkl", map_location='cpu')
        ckpt = load_dict["model_state"]
        mmp.load_state_dict(ckpt)
        list_encoded_data = load_dict["list_encoded_data"]
        best_mmp = copy.copy(mmp)
    else:
        list_encoded_data = []
        best_val_loss = torch.inf
        opt = torch.optim.Adam(mmp.parameters(), lr=0.0001)
        
        for epoch in range(1000):
            for x_c, x_nn, _ in dl:
                results = mmp.train_step(x_c.to(device), x_nn.to(device), optimizer=opt)
            
            if epoch % 10 == 0:
                val_loss = mmp.validation_step(ds.traj_data_.to(device))['loss']
                print(f"[Epoch {epoch}] Train loss: {results['loss']} Val loss: {val_loss}")
                list_encoded_data.append(mmp.encode(ds.traj_data_.to(device)).detach().cpu())
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_mmp = copy.copy(mmp)
                    print(f"best_val_loss is updated to {best_val_loss}")
        
        # torch.save({
        #     "list_encoded_data": list_encoded_data, 
        #     "model_state": best_mmp.state_dict()
        # }, "results/nrmmp.pkl")

    # Visualization functions
    def plot_epoch(epoch):
        plt.figure(figsize=(6, 6))
        for z_data, l in zip(list_encoded_data[epoch], ds.labels_):
            c = 'lightskyblue' if l[0] == 0 else 'darkmagenta'
            s = 50 if l[1] == 0 else 100
            
            if l[2] == 1:
                marker = '<'
            elif l[2] == 2:
                marker = '3'
            elif l[2] == 3:
                marker = '+'
            elif l[2] == 4:
                marker = '4'
            elif l[2] == 5:
                marker = '>'
            
            plt.scatter(z_data[0], z_data[1], s=s, c=c, marker=marker)
        
        plt.title(f"Epoch {epoch}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    # Interactive visualization loop
    print("\nVisualizing training progression:")
    for epoch in range(0, len(list_encoded_data), max(1, len(list_encoded_data)//10)):
        plot_epoch(epoch)

    # Final visualization
    print("\nFinal optimized embedding:")
    plt.figure(figsize=(6, 6))
    encoded_data = best_mmp.encode(ds.traj_data_.to(device)).detach().cpu()
    
    for z_data, l in zip(encoded_data, ds.labels_):
        c = 'lightskyblue' if l[0] == 0 else 'darkmagenta'
        s = 50 if l[1] == 0 else 100
        
        if l[2] == 1:
            marker = '<'
        elif l[2] == 2:
            marker = '3'
        elif l[2] == 3:
            marker = '+'
        elif l[2] == 4:
            marker = '4'
        elif l[2] == 5:
            marker = '>'
        
        plt.scatter(z_data[0], z_data[1], s=s, c=c, marker=marker)
    
    # Add trajectory lines
    data1 = encoded_data[ds.labels_[:, 0] == 0]
    data2 = encoded_data[ds.labels_[:, 0] == 1]
    
    for x1, x2 in zip(data1[:-1], data1[1:]):
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='lightskyblue', alpha=0.5)
    
    for x1, x2 in zip(data2[:-1], data2[1:]):
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='darkmagenta', alpha=0.5)
    
    plt.title("Final Embedding Space with Trajectories")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

if __name__ == "__main__":
    main()