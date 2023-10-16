import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# Data
def get_dataloader(
    dataset='cifar10',
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train, data_val = random_split(
        dataset=torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train),
        lengths=split,
        generator=torch.Generator().manual_seed(42)
    )
    data_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'classes': classes
    }