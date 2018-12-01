#!/usr/bin/python3
import torch
import torch.nn as nn
import sys, pickle
sys.path.append("../")
import utils
from utils import normal_init

import numpy
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

class ResNetEditor(nn.Module):
    def __init__(self, dim=24):
        super(ResNetEditor, self).__init__()
        d = dim

        self.res_block_1_d = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(), 
            nn.Conv2d(3, d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, d, kernel_size=(3,3), padding=1)
        )

        self.res_block_d_d = nn.Sequential(
            nn.BatchNorm2d(d),
            nn.ReLU(), 
            nn.Conv2d(d, d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, d, kernel_size=(3,3), padding=1)
        )

        self.res_block_d_d2 = nn.Sequential(
            nn.BatchNorm2d(d),
            nn.ReLU(), 
            nn.Conv2d(d, d*2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),
            nn.Conv2d(d*2, d*2, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d2_d2 = nn.Sequential(
            nn.BatchNorm2d(d*2),
            nn.ReLU(), 
            nn.Conv2d(d*2, d*2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),
            nn.Conv2d(d*2, d*2, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d2_d = nn.Sequential(
            nn.BatchNorm2d(d*2),
            nn.ReLU(), 
            nn.Conv2d(d*2, d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, d, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d_1 = nn.Sequential(
            nn.BatchNorm2d(d),
            nn.ReLU(), 
            nn.Conv2d(d, d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, 3, kernel_size=(3,3), padding=1)
        )
        
        self.relu = nn.Sequential(nn.ReLU())
        self.c = nn.Parameter(torch.ones(1))

    def forward(self, image):
        image = image.detach()
        image = image.view(-1,3,32,32)
        image_rd = image.repeat((1,8,1,1))
        
        x_1d = self.res_block_1_d(image)
        x_1d = x_1d + image_rd

        x_dd = self.res_block_d_d(x_1d)
        x_dd = x_dd + x_1d

        x_dd2 = self.res_block_d_d2(x_dd)
        x_dd2 = x_dd2 + x_dd.repeat((1,2,1,1))
        
        x_d2d2 = self.res_block_d2_d2(x_dd2)
        x_d2d2 = x_d2d2 + x_dd2

        x_d2d = self.res_block_d2_d(x_d2d2)
        x_d2d = x_d2d + self.res_block_d2_d(x_d2d2)

        x_d1 = self.res_block_d_1(x_d2d)
        #x_d1 = nn.functional.tanh(x_d1)
        x_d1 = self.c * image + x_d1
        
        return x_d1

def unit_test():
    test_gen = ResNetEditor()

    images = torch.rand((10, 3, 32, 32))              # Gets us 10 batches of noise (100 dim)
    output = test_gen.forward(images)
    print(output.shape)
    assert(images.shape == (10,3,32,32))
    utils.print_network(test_gen)

if __name__ == "__main__":
    unit_test()

