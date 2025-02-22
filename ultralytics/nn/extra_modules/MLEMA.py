import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch.model import MemoryEfficientSwish

import itertools
import einops
import math
import numpy as np
from einops import rearrange
from torch import Tensor
from typing import Tuple, Optional, List
from timm.models.layers import trunc_normal_


class MCCA(nn.Module):
    def __init__(self, in_size, local_size=5, local_weight=0.5, gamma = 2, b = 1):
        super(MCCA, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b 
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.local_weight = local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        local_arv = self.local_arv_pool(x) #bc55 conv2d bc55 

        #global_arv = self.global_arv_pool(local_arv)    
 

        b, c, h, w = x.shape # 2 512 20 20

        b_local, c_local, h_local, w_local = local_arv.shape # 2 512 5 5
        print(local_arv.shape,'wwwwwwwww')
        
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1) # 2 512 25       2 25 512     2 1 25*512
     
        #temp_global = global_arv.view(b, c, -1).transpose(-1, -2) # 2 512 1      2 1 512

        y_local = self.conv1(temp_local) # 2 1 12800
        print(y_local.shape,'y_local')
        #y_global = self.conv2(temp_global) # 2 1 512
        y_global = self.conv2(local_arv)
        print(y_global.shape,'ssssssss')

        att_global = self.gate(y_global)

        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        print(y_local_transpose.shape,'wwwwwwwww')
        # 2 25 512     2 512 25    2 512 5 5
        #y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)
         
        att_local = self.gate(y_local_transpose)
        print(att_local.shape,'dddddddddddd')
        #att_global = F.conv_transpose2d(self.gate(y_global_transpose),torch.ones((c, 1, self.local_size, self.local_size)), stride=self.local_size)
        
        # att_global = F.conv_transpose2d(self.gate(y_global),torch.ones((c, 1, self.local_size, self.local_size)), stride=self.local_size)
        # print(att_global.shape,'gggggggggggg')
        # att_all = F.conv_transpose2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), 
        #                               torch.ones((c, 1, int(h/self.local_size), int(w/self.local_size))), stride=int(w/self.local_size))
        att_all = F.conv_transpose2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), 
                                      torch.ones((c, 1, int(h/self.local_size), int(w/self.local_size))), stride=int(w/self.local_size))
        print(att_all.shape,'nnnnnnnnnn')
        x = x * att_all
        
        return x


if __name__ == '__main__':
    input_tensor = torch.ones((1, 512, 20, 20))
    print(input_tensor.shape)

    model = MCCA(1000)
    out = model(input_tensor)
    print(out.shape)

