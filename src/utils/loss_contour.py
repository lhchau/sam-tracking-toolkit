import os
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import loss_landscapes
import loss_landscapes.metrics

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loss_contour(
    net,
    dataloader,
    save_fig=False,
    split="train",
    name="",
    distance=10,
    step=40
):
    criterion = nn.CrossEntropyLoss()

    x, y = iter(dataloader).__next__()
    x, y = x.to(device), y.to(device)
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    model_final = copy.deepcopy(net)

    loss_data_fin = loss_landscapes.random_plane(model_final, metric, distance, step, normalization='filter', deepcopy_model=True)

    fig = plt.figure()
    plt.contour(loss_data_fin, levels=50)
    plt.title('Loss Contours around Trained Model')

    # Save the plot as an image (e.g., as a PNG file)
    if save_fig:
        directory = f"./assets/{name}/{split}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{distance}_{step}.png', dpi=300)
    return fig