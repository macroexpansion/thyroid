import os
import torch
import cv2
import numpy as np

from skimage import io


def save_incorrect_images(images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, indices: torch.Tensor):
    class_labels = ["2", "3", "4"]
    images = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    for idx in range(images.shape[0]):
        folder_name = str(class_labels[labels[idx]])
        imagename = f"{indices[idx]}_pred_{class_labels[preds[idx]]}.jpg"
        path = os.path.join("saved", folder_name, imagename)
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        io.imsave(path, images[idx])
