import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
import time

from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


from utils.misc import set_seed
from utils.training import EarlyStopping
from dataloader import dataloader, FixedSizePadding
from logs import log_writer


torch.backends.cudnn.benchmarks = True

use_gpu = torch.cuda.is_available()
device = "cuda:1" if use_gpu else "cpu"

preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def train(
    model,
    loss_fn,
    optimizer,
    lr_scheduler=None,
    num_epochs: int = 500,
    batch_size: int = 16,
    seed: int = 3,
    model_name: str = "resnet50",
):
    if use_gpu:
        print("Using CUDA")
        model.to(device)

    es = EarlyStopping(mode="max", patience=20)
    since = time.time()

    best_acc = 0.0
    rows = {"train": [], "valid": []}

    scaler = GradScaler()
    for epoch in range(1, num_epochs):
        start = time.time()
        print("-" * 10)
        print(f"Epoch {epoch}/{num_epochs}")

        # training and validateing
        dataset, data_loader, data_size = dataloader(
            path="center_crop_data", data="train", batch_size=batch_size, transform=preprocess["train"], seed=seed
        )
        for phase in ["train", "valid"]:
            print(phase)
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for images, labels, _ in tqdm(data_loader[phase]):
                images = images.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

                del images, labels, outputs, preds
                torch.cuda.empty_cache()

            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects / data_size[phase]

            rows[phase].append([epoch_loss, epoch_acc.item()])
            log_writer(
                header=["loss", "accuracy"],
                rows=rows[phase],
                folder_name=model_name,
                file_name=f"{phase}.csv",
            )

            if phase == "train":
                # save last weight
                if not os.path.exists(f"weights/{model_name}"):
                    os.makedirs(f"weights/{model_name}")
                torch.save(model.state_dict(), f"weights/{model_name}/last-{model_name}.pt")

                # early stopping
                if es.step(epoch_acc.cpu()):
                    time_elapsed = time.time() - since
                    print("Early Stopping")
                    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                    print("Best val Acc: {:4f}".format(best_acc))
                    return

            if phase == "valid":
                # learning rate scheduler
                if lr_scheduler:
                    lr_scheduler.step(epoch_loss)

                # save best weight
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print("Update best acc: {:4f}".format(best_acc))
                    torch.save(model.state_dict(), f"weights/{model_name}/best-{model_name}.pt")

            print(f"\t--> Loss: {epoch_loss}")
            print(f"\t--> Accuracy: {epoch_acc}")

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))


__all__ = ["train"]
