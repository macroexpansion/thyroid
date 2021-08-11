from typing import List, Any

import csv
import os


def log_writer(header: List[str], rows: List[List[Any]], folder_name: str = "log", file_name: str = "log.csv"):
    if not os.path.exists(os.path.join("logs", folder_name)):
        os.mkdir(os.path.join("logs", folder_name))

    path = os.path.join("logs", folder_name, file_name)
    with open(path, "w", newline="") as fo:
        writer = csv.writer(fo, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    pass
