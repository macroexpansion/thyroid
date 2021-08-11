import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def distribution_data():
    all_data = ImageFolder(root="padding_data", transform=transforms.ToTensor())
    # print(all_data.class_to_idx)
    lb = {0: "2", 1: "3", 2: "4", 3: "5"}
    sampler = SubsetRandomSampler(list(range(len(all_data))))
    dataloader = DataLoader(all_data, batch_size=1, sampler=sampler)
    data = {}
    for img, label in dataloader:
        _, h, w = img.squeeze().size()
        label = label.numpy()[0]
        if label not in data.keys():
            data[label] = []
        data[label].append([h, w])

    for key in data.keys():
        data[key] = np.array(data[key])
        print(f"max (h, w) {lb[key]}: ({np.max(data[key][:, 0])}, {np.max(data[key][:, 1])}) ")
        # print(f"max h {lb[key]}: ")
    scatter_plot(data)


def scatter_plot(data):
    lb = {0: "2", 1: "3", 2: "4", 3: "5"}
    colors = ["blue", "orange", "green", "red"]
    fig, ax = plt.subplots()
    for key in data.keys():
        h = data[key][:, 0]
        w = data[key][:, 1]
        ax.scatter(h, w, c=colors[key], label=lb[key], alpha=0.3, edgecolors="none")
    ax.set(xlabel="height", ylabel="width")
    ax.legend()
    ax.grid(True)
    # fig.savefig("data_height_width_distribution.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    distribution_data()
