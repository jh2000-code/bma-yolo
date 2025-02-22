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


class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, local_weight=0.5, gamma = 2, b = 1):
        super(MLCA, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b 
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_weight = local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        local_arv = self.local_arv_pool(x) #bc55 conv2d bc55 
        print (local_arv.shape,'local_arv.shape') 
        global_arv = self.global_arv_pool(local_arv)    
        print(global_arv.shape,'global_arv.shape')

        b, c, h, w = x.shape
        print(x.shape,'x.shape')
        b_local, c_local, h_local, w_local = local_arv.shape
        print(local_arv.shape,'local_arv.shape') #b c h*w    b h*w c   b 1 h*w*c
        
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        print(temp_local.shape,'temp_local.shape')        
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
        print(temp_global.shape,'temp_global.shape')
        
        y_local = self.conv(temp_local)
        print(y_local.shape,'y_local.shape')
        y_global = self.conv(temp_global)
        print(y_global.shape,'y_global.shape')
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        print(y_local_transpose.shape,'y_local_transpose.shape')
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)
        print(y_global_transpose.shape,'y_global_transpose.shape')
        att_local = self.gate(y_local_transpose)
        print(att_local.shape,'att_local.shape')
        att_global = F.conv_transpose2d(self.gate(y_global_transpose),torch.ones((c, 1, self.local_size, self.local_size)), stride=self.local_size)
        print(att_global.shape,'att_global.shape')
        att_all = F.conv_transpose2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), 
                                      torch.ones((c, 1, int(h/self.local_size), int(w/self.local_size))), stride=int(w/self.local_size))
        print(att_all.shape,'att_all.shape')
        x = x * att_all
        
        return x


if __name__ == '__main__':
    input_tensor = torch.ones((2, 512, 20, 20))
    print(input_tensor.shape)

    model = MLCA(20)
    out = model(input_tensor)
    print(out.shape)


# class EMA(nn.Module):
#     def __init__(self, channels, factor=8):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
#         self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         b, c, h, w = x.size()
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)