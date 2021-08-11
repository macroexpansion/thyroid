import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from skimage import io
from torchvision import models, transforms
from torch.cuda.amp import autocast
from torch.backends import cudnn

from models import resnet18
from dataloader import FixedSizePadding


use_gpu = False  # torch.cuda.is_available()
device = "cuda:0" if use_gpu else "cpu"


def inference(model, image):
    if use_gpu:
        print("Using CUDA")
        model.to(device)

    with torch.no_grad():
        with autocast():
            image = image.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
        return pred.item()


if __name__ == "__main__":
    targets = ["3", "4"]
    transforms = transforms.Compose([transforms.ToTensor(), FixedSizePadding(), lambda x: x.unsqueeze(0)])

    image = io.imread("padding_data/3/BANHTHIDUC1963_20191222_Small-_Part_0001_3_0.jpg")
    image = transforms(image)

    net = resnet18(load_weight=None)
    print(image.size())
    start = time.time()
    prediction = inference(net, image)
    print(f"inference time {device}: {time.time() - start}")
    print(targets[prediction])
