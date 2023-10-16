import torch

from src.data.get_dataloader import get_dataloader
from src.utils.get_sharpness import *
from src.models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ResNet18()
net = net.to(device)
scaler = torch.cuda.amp.GradScaler()

data_dict = get_dataloader(
    batch_size=128, 
    num_workers=4, 
    split=(0.8, 0.2)
    )

train_dataloader, _, _, _ = data_dict['train_dataloader'], data_dict['val_dataloader'], \
    data_dict['test_dataloader'], data_dict['classes']

sharpness = get_avg_sharpness(net, scaler, train_dataloader, noisy_examples='default', sigma=0.1)

print(sharpness)