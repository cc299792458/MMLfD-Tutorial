import os
import copy
import torch
import matplotlib.pyplot as plt

from models.mmp import MMP
# from loader.Pouring_dataset import Pouring
# from models.modules import FC_SE32vec, FC_vec2SE3
from loader.Toy_dataset import Toy
from models.modules import FC_traj2latent, FC_latent2traj


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    use_pretrained = os.path.exists("results/mmp.pkl")

    # ds = Pouring()
    ds = Toy()
    dl = torch.utils.data.DataLoader(ds, batch_size=10)
    
    # encoder = FC_SE32vec(
    #     in_chan=480 * 12,
    #     out_chan=2,
    #     l_hidden=[2048, 1024, 512, 256, ],
    #     activation=['gelu', 'gelu', 'gelu', 'gelu',],
    #     out_activation='linear'
    # )
    
    # decoder = FC_vec2SE3(
    #     in_chan=2,
    #     out_chan=480 * 6,
    #     l_hidden=[256, 512, 1024, 2048, ],
    #     activation=['gelu', 'gelu', 'gelu', 'gelu',],
    #     out_activation='linear'
    # )

    encoder = FC_traj2latent(
        in_chan=201*2,
        out_chan=2,
        l_hidden=[2048, 1024, 512, 256, ],
        
    )
    
    mmp = MMP(encoder, decoder)
    mmp.to(device)

    if use_pretrained:
        load_dict = torch.load("results/mmp.pkl", map_location='cpu')
        ckpt = load_dict["model_state"]
        mmp.load_state_dict(ckpt)
        list_encoded_data = load_dict["list_encoded_data"]
        best_mmp = copy.copy(mmp)
    else:
        list_encoded_data = []
        best_val_loss = torch.inf
        opt = torch.optim.Adam(mmp.parameters(), lr=0.0001)
        for epoch in range(1000):
            for x, _ in dl:
                results = mmp.train_step(x.to(device), optimizer=opt)
            if epoch%10 == 0:
                val_loss = mmp.validation_step(x.to(device))['loss']
                print(f"[Epoch {epoch}] Train loss: {results['loss']} Val loss: {val_loss}")
                list_encoded_data.append(mmp.encode(ds.traj_data_.to(device)).detach().cpu())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_mmp = copy.copy(mmp)
                    print(f"best_val_loss is updated to {best_val_loss}")
        torch.save({
            "list_encoded_data": list_encoded_data, 
            "model_state": best_mmp.state_dict()
        }, "results/mmp.pkl")

    # Plotting the final results
    plt.figure(figsize=(3, 3))
    encoded_data = best_mmp.encode(ds.traj_data_.to(device)).detach().cpu()
    
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
    
    data1 = encoded_data[ds.labels_[:, 0]==0]
    data2 = encoded_data[ds.labels_[:, 0]==1]
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()