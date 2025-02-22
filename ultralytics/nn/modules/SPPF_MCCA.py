import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock



class SPPF_MCCA(nn.Module):

    def __init__(self, c1, c2, k=5):  
        super().__init__()
        c_ = c1 // 2  
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attention = MCCA(c2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = torch.cat((x, y1, y2, self.m(y2)), 1) #2 512 20 20
        print(y3.shape,'sssssssssssssssssssssss')
        y4 = self.attention(y3)
        return self.cv2(y4)
    

if __name__ == '__main__':
    input_tensor = torch.ones((2, 512, 20, 20))
    print(input_tensor.shape)

    model = SPPF_MCCA(1024,5)
    out = model(input_tensor)
    print(out.shape)

