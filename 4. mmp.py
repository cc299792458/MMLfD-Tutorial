import os
import copy
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Union

from models.mmp import MMP
from loader.Toy_dataset import Toy
from loader.Pouring_dataset import Pouring
from models.modules import FC_SE32vec, FC_vec2SE3
from models.modules import FC_traj2latent, FC_latent2traj


def main():
    # Configuration
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    use_pretrained = os.path.exists("results/mmp.pkl")
    pouring = False  # True means Pouring, False means Toy
    
    # Initialize dataset and dataloader
    ds = Pouring() if pouring else Toy()
    dl = torch.utils.data.DataLoader(ds, batch_size=10)
    
    # Initialize model components
    if pouring:
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
    
    # Initialize model
    mmp = MMP(encoder, decoder, type_='SE3' if pouring else 'linear').to(device)
    data = ds.traj_data_ if pouring else ds.data
    
    # Training or loading pretrained model
    train_losses: List[float] = []
    val_losses: List[float] = []
    list_encoded_data: List[torch.Tensor] = []
    
    if use_pretrained:
        load_dict = torch.load("results/mmp.pkl", map_location='cpu')
        mmp.load_state_dict(load_dict["model_state"])
        list_encoded_data = load_dict["list_encoded_data"]
        best_mmp = copy.copy(mmp)
    else:
        best_val_loss = float('inf')
        optimizer = torch.optim.Adam(mmp.parameters(), lr=0.0001)
        
        for epoch in range(1000):
            # Training step
            for x, _ in dl:
                results = mmp.train_step(x.to(device), optimizer=optimizer)
                train_loss = results['loss']
            
            # Validation and logging
            if epoch % 10 == 0:
                val_results = mmp.validation_step(x.to(device))
                val_loss = val_results['loss']
                
                print(f"[Epoch {epoch}] Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}")
                
                list_encoded_data.append(mmp.encode(data.to(device)).detach().cpu())
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_mmp = copy.copy(mmp)
                    print(f"best_val_loss is updated to {best_val_loss:.4f}")
        
        # Save model (commented out as in original)
        # torch.save({
        #     "list_encoded_data": list_encoded_data, 
        #     "model_state": best_mmp.state_dict()
        # }, "results/mmp.pkl")

    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Encoded data visualization
    plt.subplot(1, 2, 2)
    encoded_data = best_mmp.encode(data.to(device)).detach().cpu()
    
    if pouring:
        for z_data, l in zip(encoded_data, ds.labels_):
            if l[0] == 0:
                c = 'lightskyblue'
            else:
                c = 'darkmagenta'
            if l[1] == 0:
                s = 50
            else:
                s = 100
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
    else:
        for z_data, l in zip(encoded_data, ds.targets):
            if l == 0:
                c = 'lightskyblue'  
            elif l == 1:
                c = 'darkmagenta'
            else:
                c = 'blue'
            plt.scatter(z_data[0], z_data[1], c=c)

    plt.title('Encoded Data Visualization')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()