import torch
import time
import numpy as np

from models import ResNet50
from utils.visualization import show_image
from utils.training import save_incorrect_images
from dataloader import dataloader, FixedSizePadding
from tqdm import tqdm
from torchvision import transforms
from dataloader import IndexedImageFolder
from logs import log_writer
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast


use_gpu = torch.cuda.is_available()
device = "cuda:1" if use_gpu else "cpu"

preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "test": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def evaluate(model, batch_size=16, seed=3, data="train"):
    model.eval()
    for param in model.parameters():
        param.grad = None

    if use_gpu:
        print("Using CUDA")
        model.to(device)

    with torch.no_grad():
        since = time.time()
        running_corrects = 0.0

        # all_indices = torch.tensor([]).type(torch.int16)
        all_predictions = torch.tensor([]).to(device).type(torch.int16)
        all_labels = torch.tensor([]).to(device).type(torch.int16)

        dataset, data_loader, data_size = dataloader(
            data=data,
            batch_size=batch_size,
            transform=preprocess[data],
            seed=seed,
            random_sample=True,
        )
        for images, labels, indices in tqdm(data_loader[data]):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

            diff = preds == labels.data
            incorrect_indices = torch.where(diff == False)[0]

            save_incorrect_images(
                images[incorrect_indices],
                labels[incorrect_indices],
                preds[incorrect_indices],
                indices[incorrect_indices],
            )

            running_corrects += torch.sum(diff)

            # all_indices = torch.cat((all_indices, indices), 0).type(torch.int16)
            all_predictions = torch.cat((all_predictions, preds), 0).type(torch.int16)
            all_labels = torch.cat((all_labels, labels), 0).type(torch.int16)

            del images, labels, outputs, preds, diff
            torch.cuda.empty_cache()

        epoch_acc = running_corrects / data_size[data]
        print(f"\t--> Accuracy: {epoch_acc}, Corrects: {running_corrects}, Datasize: {data_size[data]}")

        print(all_predictions)
        print(all_labels)

        print(
            classification_report(
                all_labels.cpu().numpy(), all_predictions.cpu().numpy(), target_names=["2", "3", "4"]
            )
        )
        time_elapsed = time.time() - since
        print("Evaluating complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    net = ResNet34(
        pretrained=False,
        mode="eval",
        load_weight="last",
        distributed=False,
        model_name="distributed_resnet50",
    )
    evaluate(net, data="test", batch_size=128)
