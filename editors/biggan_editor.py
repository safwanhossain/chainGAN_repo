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

class ResBlock(nn.Module):
    def __init__(self,num_in_channels,num_out_channels):
        super(ResBlock, self).__init__()
        self.main_module = nn.Sequential(
            nn.BatchNorm2d(num_in_channels),
            nn.ReLU(),
            nn.Conv2d(num_in_channels,num_out_channels//2,3,padding=1),
            nn.BatchNorm2d(num_out_channels//2),
            nn.ReLU(),
            nn.Conv2d(num_out_channels//2,num_out_channels,3,padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(num_in_channels,num_out_channels,1)
        )
    def forward(self, x):
        return self.main_module(x) + self.skip(x)

class BigGANEditor(nn.Module):
    def __init__(self,channel_list,rgb=False):
        super(BigGANEditor,self).__init__()
        block_list = [ResBlock(in_channel, out_channel) for in_channel, out_channel in channel_list]
        self.blocks = nn.Sequential(*block_list)
        self.critic_gate = CriticGate(rgb)
        # Initialization
        for p in self.parameters():
            if len(p.shape) >= 2:
                orth_init(p)

    def forward(self,x):
        x_prime = self.blocks(x)
        critic_val = self.critic_gate(x_prime)
        out = critic_val * x_prime + (1 - critic_val) * x
        return out
