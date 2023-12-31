'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import wandb
import yaml

from src.models import *
from src.utils.utils import progress_bar, get_mask_layers, get_prop_of_neg
from src.data.get_dataloader import get_dataloader
from src.utils.loss_landscape import get_loss_landscape
# from src.optimizer.sam import SAM 
from src.optimizer.customized_sam import SAM 
from src.utils.bypass_bn import enable_running_stats, disable_running_stats
from src.utils.get_similarity_score import get_similarity_score, get_named_parameters
from src.utils.get_sharpness import get_avg_sharpness
from src.utils.get_rank_matrix import *
from src.utils.step_lr import StepLR

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', default='example', type=str, help='path to YAML config file')
args = parser.parse_args()

with open(f"./config/{args.experiment}.yaml", "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("==> Read YAML config file successfully ...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

EPOCHS = cfg['trainer']['epochs'] 

name = cfg['wandb']['name']
# Initialize Wandb
print('==> Initialize wandb..')
wandb.init(project=cfg['wandb']['project'], name=cfg['wandb']['name'])
# define custom x axis metric
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*", step_metric="epoch")
wandb.define_metric("weight/*", step_metric="epoch")

# Data
data_dict = get_dataloader(
    batch_size=cfg['data']['batch_size'], 
    num_workers=cfg['data']['num_workers'], 
    split=cfg['data']['split']
    )

train_dataloader, val_dataloader, test_dataloader, classes = data_dict['train_dataloader'], data_dict['val_dataloader'], \
    data_dict['test_dataloader'], data_dict['classes']

# Model
print('==> Building model..')
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
named_parameters = get_named_parameters(net)
scaler = torch.cuda.amp.GradScaler()

perturbated_layers = cfg['model']['perturbated_layers']
mask_layers = get_mask_layers(net, perturbated_layers)

criterion = nn.CrossEntropyLoss()
base_optimizer = optim.SGD
optimizer = SAM(
    net.parameters(), 
    base_optimizer, 
    lr=cfg['model']['lr'], 
    momentum=cfg['model']['momentum'], 
    weight_decay=cfg['model']['weight_decay'],
    rho=cfg['model']['rho'], 
    adaptive=cfg['model']['adaptive'],
    mask_layers=mask_layers
    )
scheduler = StepLR(
    optimizer, 
    learning_rate=cfg['model']['lr'],
    total_epochs=200
    )

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        enable_running_stats(net)  # <- this is the important line
        outputs = net(inputs)
        first_loss = criterion(outputs, targets)
        first_loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # get first gradient
        first_grad = optimizer._get_grad()
        optimizer.zero_grad()
        
        disable_running_stats(net)  # <- this is the important line
        criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)
        
        # get second gradient   
        second_grad = optimizer._get_grad()
        optimizer.zero_grad()
        
        sim_score, sim_scores = get_similarity_score(first_grad, second_grad, named_parameters)

        train_loss += first_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss_mean = train_loss/(batch_idx+1)
        acc = 100.*correct/total
        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss_mean, acc, correct, total))
        
        if (batch_idx + 1) % 100 == 0:
            wandb.log({
                'sim_score': sim_score, 
                'similarity/': sim_scores,
                })
    sharpness = get_avg_sharpness(net, scaler, train_dataloader, noisy_examples='default', sigma=0.1)
    sparsity = {
        f"train_block{i}": get_feature_sparsity(train_dataloader, net, i)
        for i in [1, 2, 3, 4, 5]
    }

    wandb.log({
        'train/loss': train_loss_mean,
        'train/acc': acc,
        'train/sharpness': sharpness,
        'sparsity/': sparsity,
        'epoch': epoch
        })

def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            val_loss_mean = val_loss/(batch_idx+1)
            acc = 100.*correct/total
            progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss_mean, acc, correct, total))
            
    prop_of_neg = get_prop_of_neg(net, named_parameters)
    sharpness = get_avg_sharpness(net, scaler, val_dataloader, noisy_examples='default', sigma=0.1)

    sparsity = {
        f"val_block{i}": get_feature_sparsity(train_dataloader, net, i)
        for i in [1, 2, 3, 4, 5]
    }

    wandb.log({
        'val/loss': val_loss_mean,
        'val/acc': acc,
        'val/sharpness': sharpness,
        'sparsity/': sparsity,
        'neg_weight/': prop_of_neg
        })
    
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'loss': val_loss,
            'epoch': epoch
        }
        if not os.path.isdir(f'checkpoint/{name}'):
            os.mkdir(f'checkpoint/{name}')
        torch.save(state, f'./checkpoint/{name}/ckpt_best.pth')
        best_acc = acc
        
def test():
    # Load checkpoint.
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{name}/ckpt_best.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_landscape_fig, train_contour_fig = get_loss_landscape(net, train_dataloader, save_fig=True, split='train', name=name)
    test_landscape_fig, test_contour_fig = get_loss_landscape(net, test_dataloader, save_fig=True, split='test', name=name)
    sharpness = get_avg_sharpness(net, scaler, test_dataloader, noisy_examples='default', sigma=0.1)

    sparsity = {
        f"test_block{i}": get_feature_sparsity(train_dataloader, net, i)
        for i in [1, 2, 3, 4, 5]
    }
    
    wandb.log({
        'test/loss': test_loss/(len(test_dataloader)+1),
        'test/acc': 100.*correct/total,
        'test/sharpness': sharpness,
        'sparsity/': sparsity,
        "test/loss_landscape": wandb.Image(test_landscape_fig),
        "test/loss_contour": wandb.Image(test_contour_fig),
        "train/loss_landscape": wandb.Image(train_landscape_fig),
        "train/loss_contour": wandb.Image(train_contour_fig)
        })

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+EPOCHS):
        train(epoch)
        val(epoch)
        scheduler(epoch)
    test()
    
        

