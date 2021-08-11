import torch
import matplotlib.pyplot as plt


def show_image(image: torch.Tensor):
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.show()
