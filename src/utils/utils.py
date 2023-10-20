'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

def get_mask_layers(net, perturbated_layers):
    return [
        any(layer_name in name for layer_name in perturbated_layers)
        for name, _ in net.named_parameters()
    ]

def get_prop_of_neg(model, named_parameters):
    dic = {
        "conv": (0, 0, 0, 0), "bn": (0, 0, 0, 0), "shortcut": (0, 0, 0, 0), "linear": (0, 0, 0, 0)
    }
    
    for param, name in zip(model.parameters(), named_parameters):
        for named_layer in dic.keys():
            if named_layer in name and 'weight' in name:
                dic[named_layer][0] += (param.data.view(-1) <= 0).sum().item() / len(param.data.view(-1))
                dic[named_layer][1] += 1
                break
            if named_layer in name and 'bias' in name:
                dic[named_layer][2] += (param.data.view(-1) <= 0).sum().item() / len(param.data.view(-1))
                dic[named_layer][3] += 1
                break
            
    return {
        "conv_weight": dic['conv'][0] / dic['conv'][1],
        "conv_bias": dic['conv'][2] / dic['conv'][3],
        "bn_weight": dic['bn'][0] / dic['bn'][1],
        "bn_bias": dic['bn'][2] / dic['bn'][3],
        "shortcut_weight": dic['shortcut'][0] / dic['shortcut'][1],
        "shortcut_bias": dic['shortcut'][2] / dic['shortcut'][3],
        "linear_weight": dic['linear'][0] / dic['linear'][1],
        "linear_bias": dic['linear'][2] / dic['linear'][3]
    }
            
def count_range_weights(model):  
        ranges = [1e-12, 1e-8, 1e-4, 1e-2, 1]
        counts = [0] * len(ranges)
        for param in model.parameters():  
            if param is not None:  
                param_values = param.data.view(-1).abs()
                for i in range(len(ranges)):
                    counts[i] += (param_values <= ranges[i]).sum().item()
        
        for i in range(len(ranges) - 1, 0, -1):
            counts[i] -= counts[i-1]
        return {
            "1e-12_count": counts[0],
            "1e-08_count": counts[1],
            "1e-04_count": counts[2],
            "1e-02_count": counts[3],
            "1e-00_count": counts[4]
        }

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
