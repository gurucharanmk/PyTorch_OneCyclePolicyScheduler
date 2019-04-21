import math

import numpy as np
#import torch
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.optim.optimizer import Optimizer


class OneCyclePolicyScheduler(object):
    def __init__(
            self,
            optimizer,
            max_lr,
            total_batch_size,
            max_momentum=0.95,
            min_momentum=0.8):
        self.optimizer = optimizer
        self.total_batch_size = total_batch_size
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        # Pre-computed learning rate and momentum will give training time
        # advantage
        self.lrs = np.concatenate([
            np.linspace(0.1*max_lr, max_lr, int(total_batch_size * 0.4)),
            np.linspace(max_lr, 0.1*max_lr, int(total_batch_size * 0.4)),
            np.linspace(0.1*max_lr, 0.01*max_lr,
             int(total_batch_size - np.floor(total_batch_size*0.8))),
        ])
        self.momentums = np.concatenate([
            np.linspace(max_momentum, min_momentum, int(total_batch_size * 0.4)),
            np.linspace(min_momentum, max_momentum, int(total_batch_size * 0.4)),
            np.linspace(max_momentum, max_momentum,
             int(total_batch_size - np.floor(total_batch_size * 0.8))),
        ])
        self.batch_count = -1

    def step(self):
        self.batch_count += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lrs[self.batch_count]
            param_group['momentum'] = self.momentums[self.batch_count]
