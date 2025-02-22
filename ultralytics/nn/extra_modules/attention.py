import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision

import itertools
import einops
import math
import numpy as np
from einops import rearrange
from torch import Tensor
from typing import Tuple, Optional, List
from ..modules.conv import Conv, autopad

__all__ = ['MCDA']

class MCDA(nn.Module):
    #多维卷积反池化注意力机制 Multidimensional convolutional depooling attention mechanism
    def __init__(self, in_size, local_size=5, local_weight=0.5, gamma = 2, b = 1):
        super(MCDA, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b 
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv3 = F.conv_transpose2d
        self.local_weight = local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        local_arv = self.local_arv_pool(x) 
        b, c, h, w = x.shape 
        b_local, c_local, h_local, w_local = local_arv.shape 
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1) 
        y_local = self.conv1(temp_local) 
        y_global = self.conv2(local_arv)
        att_global = self.gate(y_global)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        att_local = self.gate(y_local_transpose)
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [h, w])
        x = x * att_all
        return x