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
    dataloader
):
    # Model
    MODEL = "ResNet18"
    print(f'==> Building model {MODEL}..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_best_sam.pth')
    net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()

    DISTANCES = 1
    STEPS = 100

    x, y = iter(dataloader).__next__()
    x, y = x.to(device), y.to(device)
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    model_final = copy.deepcopy(net)

    for dist in range(10, 100 + 1, 10):
        for step in range(10, 100 + 1, 10):
            directory = f"./assets/SAM_{MODEL}_{dist}_{step}"
            if os.path.exists(directory):
                continue
            loss_data_fin = loss_landscapes.random_plane(model_final, metric, dist, step, normalization='filter', deepcopy_model=True)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X = np.array([[j for j in range(step)] for i in range(step)])
            Y = np.array([[i for _ in range(step)] for i in range(step)])
            ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_title('Surface Plot of Loss Landscape')

            # Save the plot as an image (e.g., as a PNG file)
            if not os.path.exists(directory):
                os.makedirs(directory)
                plt.savefig(f'{directory}/loss_surface_plot.png', dpi=300)  # Change the filename and format as needed
            plt.close()