from typing import List, Any

import csv
import os
import numpy as np


def log_reader(folder_name: str):
    folder_path = os.path.join("logs", folder_name)
    if not os.path.exists(folder_path):
        raise Exception("Folder not found")

    result = {"path": folder_path}
    for phase in ["train", "valid"]:
        path = os.path.join(folder_path, phase + ".csv")
        with open(path, newline="") as fo:
            reader = csv.reader(fo, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            header = next(reader)
            result[phase] = {metric: [] for metric in header}
            for row in reader:
                for idx, value in enumerate(row):
                    result[phase][header[idx]].append(np.array(value).astype(np.float32))
    return result
