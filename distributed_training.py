import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import os
import time
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from torchvision import transforms
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from utils.misc import set_seed
from utils.training import save_incorrect_images
from utils.loss import CrossEntropyLossWithLabelSmoothing
from dataloader import FixedSizePadding, distributed_dataloader
from logs import log_plot, log_writer
from models import resnet50, resnet34, resnet18, mobilenetv2
from torch.utils.tensorboard import SummaryWriter


preprocess = {
    "train": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "valid": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
    "test": transforms.Compose([transforms.ToTensor(), FixedSizePadding()]),
}


def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def distributed_train(world_size: int = 2) -> None:
    model = resnet18(out=2)
    # model = mobilenetv2(pretrained=False, out=2)
    loss_fn = CrossEntropyLossWithLabelSmoothing(classes=2, epsilon=0.05)

    mp.spawn(
        _distributed_train,
        args=(world_size, model, loss_fn),
        nprocs=world_size,
        join=True,
    )


def _distributed_train(
    rank: int,
    world_size: int,
    model: nn.Module,
    loss_fn: nn.Module,
) -> None:
    """
    train function
    """
    cudnn.enabled = True
    cudnn.benchmark = True
    setup(rank, world_size)

    params = {
        "batch_size": 384,
        "num_epochs": 200,
        "seed": 1000,
    }
    optimizer_params = {"weight_decay": 0.5, "lr": 5e-4}

    set_seed(params["seed"])
    model = model.to(rank)
    loss_fn = loss_fn.to(rank)
    scaler = GradScaler()
    model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(model.parameters(), **optimizer_param)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, threshold=1e-5, verbose=True, patience=7
    )

    if rank == 0:
        model_name = "5_resnet18"
        for key, value in params.items():
            model_name += f"_{value}{key.replace('_', '')}"

        best_acc = 0.0
        rows = {"train": [], "valid": []}
        start = time.time()
        writer = SummaryWriter(f"runs/{model_name}")

    _, data_loader, data_size = distributed_dataloader(
        rank,
        world_size,
        path="padding_data",
        data="train",
        batch_size=params["batch_size"],
        transform=preprocess["train"],
        seed=params["seed"],
    )
    for epoch in range(1, params["num_epochs"] + 1):
        if rank == 0:
            print("-" * 10)
            print(f"Epoch {epoch}/{params['num_epochs']}")

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            running_preds = torch.tensor([]).to(rank)
            running_true = torch.tensor([]).to(rank)

            for images, labels, indices in tqdm(data_loader[phase]):
                labels = labels.to(rank)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, labels)

                        if phase == "train":
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        # if phase == "valid":
                        #     if epoch % 5 == 0 and epoch > 40:
                        #         diff = preds == labels
                        #         incorrect_indices = torch.where(diff == False)[0]

                        #         save_incorrect_images(
                        #             images[incorrect_indices],
                        #             labels[incorrect_indices],
                        #             preds[incorrect_indices],
                        #             indices[incorrect_indices],
                        #         )

                # accumulate metrics
                running_preds = torch.cat((running_preds, preds)).type(torch.int8)
                running_true = torch.cat((running_true, labels)).type(torch.int8)
                running_loss += loss * images.size(0)
                running_corrects += torch.sum(preds == labels)

                del images, labels, outputs, preds, _
                torch.cuda.empty_cache()

            # gather, reduce values from 2 processes
            all_preds = [torch.empty_like(running_preds).to(rank) for _ in range(world_size)]
            all_true = [torch.empty_like(running_true).to(rank) for _ in range(world_size)]
            dist.all_gather(all_preds, running_preds)
            dist.all_gather(all_true, running_true)
            dist.reduce(running_loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(running_corrects, 0, op=dist.ReduceOp.SUM)

            # only run on first gpu
            if rank == 0:
                epoch_loss = running_loss.item() / data_size[phase]
                epoch_acc = running_corrects.item() / data_size[phase]
                all_true = torch.cat(all_true).detach().cpu().numpy()
                all_preds = torch.cat(all_preds).detach().cpu().numpy()
                precision, recall, fscore, _ = precision_recall_fscore_support(
                    all_true, all_preds, average="macro", zero_division=1
                )
                class_precision, class_recall, class_fscore, class_support = precision_recall_fscore_support(
                    all_true, all_preds, average=None, labels=[0, 1], zero_division=1
                )
                write_tensorboard(
                    writer,
                    epoch,
                    phase,
                    loss=epoch_loss,
                    accuracy=epoch_acc,
                    precision=scalars_layout(precision, class_precision),
                    recall=scalars_layout(recall, class_recall),
                    f1=scalars_layout(fscore, class_fscore),
                )
                # print(f"{running_corrects.item()} vs {data_size[phase]}")

                if phase == "train":
                    # learning rate scheduling
                    lr_scheduler.step(epoch_loss)
                    save_last_weight(model, model_name, epoch)

                elif phase == "valid":
                    # save best weight
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        save_best_weight(model, model_name, best_acc)

                logging(phase, epoch_loss, epoch_acc)
                print(
                    classification_report(
                        all_true, all_preds, target_names=["tirad 3", "tirad 4"], zero_division=1
                    )
                )

    if rank == 0:
        time_elapsed = time.time() - start
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        writer.close()

    cleanup()


def write_log_to_csv(rows: dict, phase: str, model_name: str, **kwargs) -> None:
    """
    write logs to csv
    """
    rows[phase].append([*kwargs.values()])
    log_writer(
        header=[*kwargs],
        rows=rows[phase],
        folder_name=model_name,
        file_name=f"{phase}.csv",
    )


def scalars_layout(metric, class_metric):
    classes = ["tirad 3", "tirad 4"]
    tmp = {}
    for idx, item in enumerate(classes):
        tmp.update({classes[idx]: class_metric[idx]})
    return {"macro": metric, **tmp}


def write_tensorboard(writer, epoch, phase, **kwargs):
    for key, value in kwargs.items():
        if type(value) == dict:
            writer.add_scalars(f"{key}/{phase}", value, epoch)
        else:
            writer.add_scalar(f"{key}/{phase}", value, epoch)


def logging(phase: str, epoch_loss: float, epoch_acc: float) -> None:
    """
    logging
    """
    print(f"--> {phase} loss:     {epoch_loss}")
    print(f"--> {phase} accuracy: {epoch_acc}")


def save_last_weight(model, model_name, epoch):
    """
    save last weight
    """
    if not os.path.exists(f"weights/{model_name}"):
        os.makedirs(f"weights/{model_name}")

    if epoch == 200:
        torch.save(model.state_dict(), f"weights/{model_name}/last-{model_name}.pt")


def save_best_weight(model, model_name, best_acc):
    """
    save best weight
    """
    print("Updated best acc: {:4f}".format(best_acc))
    torch.save(model.state_dict(), f"weights/{model_name}/best-{model_name}.pt")


if __name__ == "__main__":
    distributed_train()
