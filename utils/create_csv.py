import csv
import os
import re


def create_csv() -> None:
    f0 = "datasets/"
    fold0 = os.listdir(f0)

    with open("patient.csv", "w") as pat:
        pat_writer = csv.writer(pat, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        pat_writer.writerow(["name_file", "name_folder", "fna", "tirad"])

        for root, dirs, files in os.walk(f0, topdown=True):
            if ".exports" in root:
                continue
            if "tirad" in root.lower():
                regex = re.search(r"[T|t]irads.*(\d{1}\w{0,1}).+", root)
                if regex == None:
                    print(root)
                    if "IVa" in root:
                        tirad = "4a"
                else:
                    tirad = regex.group(1)
                # print(tirad)

                if "lanh" in root or "lành" in root:
                    fna = 0
                elif "ac" in root:
                    fna = 1
                else:
                    fna = 2

                for file_name in files:
                    a = [file_name, root, fna, tirad]
                    print(a)
                    pat_writer.writerow(a)


__all_ = ["create_csv"]
