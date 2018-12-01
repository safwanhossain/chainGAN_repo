#!/usr/bin/python3
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from utils import normal_init
import utils

class DCGAN_Discriminator(nn.Module):
    def __init__(self, edit_num, dim=128):
        super(DCGAN_Discriminator, self).__init__()
        self.num_last_layers = 1 + edit_num
        self.last_layers = nn.ModuleList([])

        # ====== LAYERS ======
        d = dim
        self.conv1 = nn.Sequential(
            # Conv1 - Go from 32x32 to 16x16
            nn.Conv2d(3, d, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )   
        
        self.conv2 = nn.Sequential(
            # Conv2 - Go from 15x15 to 8x8
            nn.Conv2d(d, d*2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU()
        )
        
        self.conv3 = nn.Sequential(
            # Conv3 - Go from 8x8 to 5x5
            nn.Conv2d(d*2, d*4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2)
        )

        for i in range(self.num_last_layers):
            layer = nn.Sequential(nn.Linear(4*4*4*d,1))
            self.last_layers.append(layer)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, image, edit_num, label = None):
        batch_n = image.shape[0]
        image = image.view(-1, 3, 32, 32)
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1,4*4*4*128)
        x = self.last_layers[edit_num](x)
        x = x.view(batch_n, 1)
        return x

def unit_test():
    # Some sanity checks for the generator
    test_dis = DCGAN_Discriminator(5, dim=128)

    image = torch.rand((10,3,32,32))
    image = image.view(-1,3,32,32)

    labels = torch.rand((10, 10))              # Gets us 10 batches of labels (10 dim)
    labels = labels.view(-1, 1, 1, 10)         # reshape it to get the dimensions on the channels

    output = test_dis.forward(image, 3)
    print(output.shape)
    assert(output.shape == (10,1))
    utils.print_network(test_dis)

if __name__ == "__main__":
    unit_test()

