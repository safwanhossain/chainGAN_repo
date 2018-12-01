# Import upper directories

import sys
sys.path.append("..")

# Residual generator based on Improved Training of Wasserstein GANs

# Imports:

import torch
import torch.nn as nn
import utils

# Class:

class resnet_generator(nn.Module):
    def __init__(self, d=128):
        super(resnet_generator, self).__init__()
        self.d = d
        # ====== LAYERS ======
        self.lin = nn.Sequential(
            nn.Linear(128, self.d*4*4)
        )
        self.resBlock1 = nn.Sequential(
            # Res block 1
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            )
        self.shortCut1 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(self.d, self.d, kernel_size = 1)
        )
        self.resBlock2 = nn.Sequential(
            # Res block 2
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            )
        self.shortCut2 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(self.d, self.d, kernel_size = 1)
        )
        self.resBlock3 = nn.Sequential(
            # Res block 3
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            )
        self.shortCut3 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(self.d, self.d, kernel_size = 1)
        )
        self.last = nn.Sequential(
            # Last conv
            nn.Conv2d(self.d, 3, kernel_size = (3, 3), padding = 1)
            )
    
    def forward(self, z):
        out_pre = self.lin(z).view(-1, self.d, 4, 4)
        out = self.resBlock1(out_pre)
        out = out + self.shortCut1(out_pre)
        out_pre = out
        out = self.resBlock2(out)
        out = out + self.shortCut2(out_pre)
        out_pre = out
        out = self.resBlock3(out)
        out = out + self.shortCut3(out_pre)
        out = self.last(out)
        return out

def unit_test():
    test_gen = resnet_generator(64)
    
    z = torch.randn((10, 128))
    output = test_gen(z)
    print(output.shape)
    assert(output.shape == (10,3,32,32))
    utils.print_network(test_gen)
    
if __name__ == '__main__':
    unit_test()
