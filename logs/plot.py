import matplotlib.pyplot as plt
import numpy as np
import os

from .reader import log_reader


def log_plot(data: dict = None, folder_name: str = None, dpi="figure"):
    if not data and not folder_name:
        raise Exception("At least data or folder_name must be provided")

    if folder_name:
        data = log_reader(folder_name=folder_name)

    headers = list(data["train"].keys())
    fig, (axs) = plt.subplots(1, len(headers))

    for idx, ax in enumerate(axs):
        for phase in ["train", "valid"]:
            ax.plot(range(len(data[phase][headers[idx]])), data[phase][headers[idx]], label=phase)
        ax.set(xlabel="epochs", ylabel=headers[idx])
        ax.set_title(headers[idx])
        ax.legend()
        ax.grid()
    fig.set_size_inches(10, 6)
    fig.savefig(os.path.join(data["path"], "plot.png"), dpi=dpi)
    plt.close("all")
    # plt.show()
