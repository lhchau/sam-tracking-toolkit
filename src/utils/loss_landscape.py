import os
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import loss_landscapes
import loss_landscapes.metrics

import torch.backends.cudnn as cudnn

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loss_landscape(
    net,
    dataloader,
):
    criterion = nn.CrossEntropyLoss()

    DISTANCES = 200
    STEPS = 400

    x, y = iter(dataloader).__next__()
    x, y = x.to(device), y.to(device)
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    model_final = copy.deepcopy(net)

    # directory = f"./assets/SAM_{MODEL}_{DISTANCES}_{STEPS}"
    loss_data_fin = loss_landscapes.random_plane(model_final, metric, DISTANCES, STEPS, normalization='filter', deepcopy_model=True)

    landscape_fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    
    contour_fig = plt.figure()
    plt.contour(loss_data_fin, levels=50)
    plt.title('Loss Contours around Trained Model')

    # Save the plot as an image (e.g., as a PNG file)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #     plt.savefig(f'{directory}/loss_surface_plot.png', dpi=300)  
    return landscape_fig, contour_fig