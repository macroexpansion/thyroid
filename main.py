from utils.misc import set_seed
from train import train

from models import ResNet50

import torch
import torch.nn as nn
import torch.optim as optim


def train_resnet():
    net = ResNet50(pretrained=False, load_weight=None)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=5e-4)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, verbose=True)
    print(optimizer)
    # train(
    #     model=net,
    #     loss_fn=criterion,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     num_epochs=1000,
    #     batch_size=128,
    #     seed=3,
    #     model_name="resnet50_2",
    # )


if __name__ == "__main__":
    train_resnet()
