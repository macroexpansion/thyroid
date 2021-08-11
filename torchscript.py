import torch
import torchvision
import numpy as np

from time import perf_counter
from models import resnet18
from torch.backends import cudnn

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def timer(f, *args):
    start = perf_counter()
    with torch.no_grad():
        f(*args)
    return perf_counter() - start


if __name__ == "__main__":
    if torch.cuda.is_available():
        print('Using CUDA')

    net = resnet18()
    net.eval()
    example = torch.rand(1, 3, 258, 366)
    with torch.no_grad():
        # cpu_traced_module = torch.jit.trace(net, example)
        # cpu_traced_module.save("torchscript_module/cpu_traced_tirad_model.pt")

        gpu_traced_module = torch.jit.trace(net.to(device), example.to(device))
        gpu_traced_module.save("torchscript_module/gpu_traced_tirad_model.pt")

    # model_ft = resnet18()
    # model_ft.eval()
    # x_ft = torch.rand(1,3, 258,366)
    # print(np.mean([timer(model_ft, x_ft) for _ in range(10)]))
