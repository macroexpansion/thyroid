from typing import List
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
import os, re


def resize_pad(blob: np.ndarray, img_size: int):
    high, width = blob.shape[:2]
    if high >= width:
        blob = cv2.resize(blob, (int(img_size * width / high), img_size))
        delta = img_size - img_size * width / high
        left, right = int(delta // 2), int(delta // 2)
        top, bottom = 0, 0
    else:
        blob = cv2.resize(blob, (img_size, int(img_size * high / width)))
        delta = img_size - img_size * high / width
        top, bottom = int(delta // 2), int(delta // 2)
        left, right = 0, 0
    # imgs.append(blob)
    new_im = cv2.copyMakeBorder(blob, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    new_im = cv2.resize(new_im, (img_size, img_size))
    return new_im


def get_labels(image_path: str) -> [str, str]:
    regex = re.search(r"\/[T|t]irads?.*(\d{1}).+\/", image_path)
    if regex == None:
        if "IVa" in image_path:
            tirad = "4"
        try:
            if tirad == "4":
                pass
        except UnboundLocalError:
            raise UnboundLocalError
    else:
        tirad = regex.group(1)

    if "lanh" in image_path or "lành" in image_path:
        fna = 0
    elif "ac" in image_path:
        fna = 1
    else:
        fna = 2

    return str(tirad), str(fna)


def get_contour(contour_path: List[list]) -> np.ndarray:
    # print(contour_path)
    contour = []
    for i in range(len(contour_path) // 2):
        x = int(contour_path[2 * i])
        y = int(contour_path[2 * i + 1])
        contour.append([[x, y]])
    contour = np.asarray(contour)
    return contour


def draw_segment(img: np.ndarray, contour: np.ndarray) -> None:
    cv2.drawContours(img, [contour], 0, 255, 1)


def crop_contour(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    height, width, channel = img.shape
    x, y, w, h = cv2.boundingRect(contour)
    pad = 10
    crop = img[max(y - pad, 0) : min(y + h + pad, height), max(x - pad, 0) : min(x + w + pad, width)]
    # crop = resize_pad(crop, 96)
    return crop


def save_image(img: np.ndarray, image_name: str, tirad: str, fna: str, save_path: str) -> None:
    if save_path is None:
        raise Error("path not found")

    ouput_folder = f"{save_path}/{tirad}"
    if not os.path.exists(ouput_folder):
        os.makedirs(ouput_folder)
    new_image_name = image_name.replace(".BMP", "") + "_".join(["", tirad, fna]) + ".jpg"
    cv2.imwrite(os.path.join(ouput_folder, new_image_name), img)


def center_crop(img: np.ndarray, contour: np.ndarray, x_margin: int = 183, y_margin: int = 129) -> np.ndarray:
    height, width, _ = img.shape
    x, y, w, h = cv2.boundingRect(contour)
    x_center, y_center = int(x + w / 2), int(y + h / 2)
    crop = img[
        max(y_center - y_margin, 0) : min(y_center + y_margin, height),
        max(x_center - x_margin, 0) : min(x_center + x_margin, width),
    ]
    return crop


def multisize_crop(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    height, width, _ = img.shape
    # if height <= 150 and width <= 250:
    #     crop = center_crop(img, contour, x_margin=125, y_margin=75)
    # elif height > 150 and width > 250:


def create_data_images():
    folders = os.listdir("datasets")
    i = 0
    for folder in folders:
        export_folder = os.path.join("datasets", folder, folder + ".exports")
        json_file = os.path.join(export_folder, os.listdir(export_folder)[0])
        with open(json_file) as fo:
            data = json.load(fo)
        images, annotations = data["images"], data["annotations"]
        for root, dirs, files in os.walk(os.path.join("datasets", folder)):
            if "irad" not in root:
                continue
            for image_name in files:
                # print(image_name)
                for image in images:
                    if image_name == image["file_name"]:
                        image_id = image["id"]
                        break

                image_path = os.path.join(root, image_name)
                if not os.path.exists(image_path):
                    print(image_path)
                    continue
                try:
                    tirad, fna = get_labels(image_path)
                except UnboundLocalError:
                    print(f"unbound {image_path}")
                    continue

                for annotation in annotations:
                    if image_id == annotation["image_id"]:
                        img = cv2.imread(image_path)
                        segmentations = annotation["segmentation"]
                        max_contour_idx = np.argmax(
                            [len(contour_path) for contour_path in segmentations]
                        )  # find contour with longest length
                        contour_path = segmentations[max_contour_idx]
                        contour = get_contour(contour_path)
                        # draw_segment(img, contour)
                        try:
                            # img = crop_contour(img, contour)
                            img = center_crop(img, contour, x_margin=125, y_margin=75)
                            # plt.imshow(img)
                            # plt.show()
                            # print(img.shape)
                        except:
                            print(image_path)
                            break
                        save_image(img, image_name, tirad, fna, "center_crop_data_2")
                        break
        #         break
        #     break
        # break


__all__ = ["create_data_images"]
