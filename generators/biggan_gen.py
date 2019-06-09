import torch
import torch.nn as nn
from torch.nn.init import orthogonal_ as orth_init

class CriticGate(nn.Module):
    def __init__(self,rgb=False):
        super(CriticGate,self).__init__()
        in_channel = 3 if rgb else 1
        self.gate_calc = nn.Sequential(
            nn.Conv2d(in_channel,1,10),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch_size = x.shape[0]
        calculated = self.gate_calc(x).view(batch_size,-1)
        avg = calculated.mean(dim=1)
        return avg

class ResBlockUp(nn.Module):
    def __init__(self,num_in_channels,num_out_channels,scale):
        super(ResBlockUp, self).__init__()
        self.main_module = nn.Sequential(
            nn.BatchNorm2d(num_in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=scale),
            nn.Conv2d(num_in_channels,num_out_channels//2,3,padding=1),
            nn.BatchNorm2d(num_out_channels//2),
            nn.ReLU(),
            nn.Conv2d(num_out_channels//2,num_out_channels,3,padding=1)
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=scale),
            nn.Conv2d(num_in_channels,num_out_channels,1)
        )
    def forward(self, x):
        return self.main_module(x) + self.skip(x)

class biggan_gen(nn.Module):
    def __init__(self,channel_list,end_pic_dim=28,z_dim=128,rgb=False):
        super(biggan_gen,self).__init__()
        block_list = [ResBlockUp(in_channel, out_channel, scale) for in_channel, out_channel, scale in channel_list]
        self.blocks = nn.Sequential(*block_list)
        begin_dim = end_pic_dim
        for _,_,scale in channel_list:
            begin_dim /= scale
        assert(int(begin_dim)==begin_dim)
        self.shape = (channel_list[0][0],int(begin_dim),int(begin_dim))
        
        self.fc = nn.Sequential(
            nn.Linear(z_dim,self.shape[0]*self.shape[1]*self.shape[2]),
            nn.BatchNorm1d(self.shape[0]*self.shape[1]*self.shape[2]),
            nn.ReLU()
        )
        
        # Initialization
        for p in self.parameters():
            if len(p.shape) >= 2:
                orth_init(p)

    def forward(self,x):
        x = self.fc(x).view(-1,*self.shape)
        x = self.blocks(x)
        return x
