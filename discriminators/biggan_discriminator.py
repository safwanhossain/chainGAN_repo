import torch
import torch.nn as nn
from torch.nn.init import orthogonal_ as orth_init

class ResBlockDown(nn.Module):
    def __init__(self,num_in_channels,num_out_channels,scale):
        super(ResBlockDown, self).__init__()
        self.main_module = nn.Sequential(
            nn.BatchNorm2d(num_in_channels),
            nn.ReLU(),
            nn.Conv2d(num_in_channels,num_out_channels//2,3,stride=scale,padding=1),
            nn.BatchNorm2d(num_out_channels//2),
            nn.ReLU(),
            nn.Conv2d(num_out_channels//2,num_out_channels,3,padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(num_in_channels,num_out_channels,1,stride=scale)
        )
    def forward(self, x):
        return self.main_module(x) + self.skip(x)

class biggan_discriminator(nn.Module):
    def __init__(self,channel_list,editor_num,begin_pic_dim=28,rgb=False):
        super(biggan_discriminator,self).__init__()
        block_list = [ResBlockDown(in_channel, out_channel, scale) for in_channel, out_channel, scale in channel_list]
        self.blocks = nn.Sequential(*block_list)
        end_dim = begin_pic_dim
        for _,_,scale in channel_list:
            end_dim /= scale
        assert(int(end_dim)==end_dim)
        self.shape = channel_list[-1][1]*int(end_dim)*int(end_dim)
        self.fc_list = nn.ModuleList([nn.Linear(self.shape,1) for i in range(editor_num+1)])
        
        # Initialization
        for p in self.parameters():
            if len(p.shape) >= 2:
                orth_init(p)

    def forward(self,x,edit_num):
        x = self.blocks(x)
        score = self.fc_list[edit_num](x.view(-1,self.shape))
        return score
