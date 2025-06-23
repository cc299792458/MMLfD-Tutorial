import matplotlib.pyplot as plt

from loader.Toy_dataset import Toy, toy_visualizer
from loader.Pouring_dataset import Pouring

ds1 = Toy('datasets/EXP1/')
ds2 = Toy('datasets/EXP2/')
ds3 = Toy('datasets/EXP3/')

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
toy_visualizer(ds1.env, axs[0], traj=ds1.data, label=ds1.targets)
toy_visualizer(ds2.env, axs[1], traj=ds2.data, label=ds2.targets)
toy_visualizer(ds3.env, axs[2], traj=ds3.data, label=ds3.targets)

ds = Pouring()
print(ds.traj_data_.size())
print(ds.labels_.size())