'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import wandb

from src.models import *
from src.utils.utils import progress_bar, count_range_weights
from src.data.get_dataloader import get_dataloader
from src.utils.loss_landscape import get_loss_landscape
from src.optimizer.sam import SAM 
from src.utils.bypass_bn import enable_running_stats, disable_running_stats


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

EPOCHS = 200

# Initialize Wandb
name = "ex04_L2_bs-128_rho-0.05_momen-0.9_ASAM_wd-5e-4"
print('==> Initialize wandb..')
wandb.init(project="Task 1 - Rerun", name=name)

# Data
data_dict = get_dataloader(batch_size=128, num_workers=4, split=(0.8, 0.2))
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

criterion = nn.CrossEntropyLoss()
base_optimizer = optim.SGD
optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=0.9, weight_decay=5e-4, rho=0.05, adaptive=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=EPOCHS)


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
        
        disable_running_stats(net)  # <- this is the important line
        criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)

        train_loss += first_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    wandb.log({
        'train/loss': train_loss/(len(train_dataloader)+1),
        'train/acc': 100.*correct/total
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

            progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    wandb.log({
        'val/loss': val_loss/(len(val_dataloader)+1),
        'val/acc': 100.*correct/total
        })
    
    r1_count, r2_count, r3_count, r4_count, r5_count = count_range_weights(net)
    wandb.log({
        "1e-12_count": r1_count,
        "1e-08_count": r2_count - r1_count,
        "1e-04_count": r3_count - r2_count - r1_count,
        "1e-02_count": r4_count - r3_count - r2_count - r1_count,
        "1e-00_count": r5_count - r4_count - r3_count - r2_count - r1_count
        })

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'loss': val_loss,
            'epoch': epoch,
            "1e-12_count": r1_count,
            "1e-08_count": r2_count - r1_count,
            "1e-04_count": r3_count - r2_count - r1_count,
            "1e-02_count": r4_count - r3_count - r2_count - r1_count,
            "1e-00_count": r5_count - r4_count - r3_count - r2_count - r1_count
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
    data = [[key, value] for key, value in checkpoint.items() if key[-5:] == "count"]
    table = wandb.Table(data=data, columns = ["label", "value"])
    train_fig = get_loss_landscape(net, train_dataloader)
    test_fig = get_loss_landscape(net, test_dataloader)
    
    wandb.log({
        'test/loss': test_loss/(len(test_dataloader)+1),
        'test/acc': 100.*correct/total,
        "test/weight_bar_chart": wandb.plot.bar(table, "label", "value", title="Bar Chart of Weight Range"),
        "test/loss_landscape": wandb.Image(test_fig),
        "train/loss_landscape": wandb.Image(train_fig)
        })

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+EPOCHS):
        train(epoch)
        val(epoch)
        scheduler.step()
    test()
    
        

