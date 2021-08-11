import torch
import onnx

from models import resnet18

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Using CUDA')

    net = resnet18()
    net.eval()
    net.to(device)
    input = torch.rand(1, 3, 258, 366).to(device)
    # with torch.no_grad():
        # torch.onnx.export(net, input, 'onnx/tirad_model.onnx', input_names=['input'], output_names=['output'], export_params=True)

    # onnx_model = onnx.load('onnx/tirad_model.onnx')
    # onnx.checker.check_model(onnx_model)
