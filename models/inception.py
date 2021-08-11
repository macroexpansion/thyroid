import torch
import torch.nn as nn

from torchvision import models

device = "cuda:1" if torch.cuda.is_available() else "cpu"

__all__ = ["inceptionv3"]


def inceptionv3(
    out=2, pretrained=False, mode="eval", load_weight=None, distributed=False, model_name="inceptionv3"
):
    if pretrained and load_weight:
        raise Exception("load_weight must be None when pretrained is True")

    net = models.inception_v3(pretrained=pretrained)
    net.fc = nn.Linear(in_features=2048, out_features=out)

    if load_weight in ["best", "last"]:
        state_dict = get_weight(net, f"weights/{model_name}/{load_weight}-{model_name}.pt", distributed=False)
        net.load_state_dict(state_dict)

    if mode == "eval":
        net.eval()
        for param in net.parameters():
            param.grad = None

    return net


if __name__ == "__main__":
    net = inceptionv3()
    print(net)
