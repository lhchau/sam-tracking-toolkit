import torch

checkpoint = torch.load('./checkpoint/ex01_L2_bs-128_rho-0.05_momen-0.9_SGD_no-wd/ckpt_best.pth')

print(checkpoint['1e-12_count'])