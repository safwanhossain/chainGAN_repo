# Import upper directories

import sys
sys.path.append("..")

#!/usr/bin/python3
import torch
import torch.nn as nn
import pickle
import utils
from utils import normal_init

import numpy
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

class Resnet_Editor(nn.Module):
    def __init__(self, d):
        super(Resnet_Editor, self).__init__()
        self.d = d
        assert self.d > 0, "Resnet dimension must be greater than zero. Sorry!"
        assert self.d % 3 == 0, "Resnet dimension must be divisible by 3. Sorry!"
        self.res_block_1_d = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1)
        )

        self.res_block_d_d = nn.Sequential(
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1)
        )

        self.res_block_d_d2 = nn.Sequential(
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d*2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d*2),
            nn.ReLU(),
            nn.Conv2d(self.d*2, self.d*2, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d2_d2 = nn.Sequential(
            nn.BatchNorm2d(self.d*2),
            nn.ReLU(),
            nn.Conv2d(self.d*2, self.d*2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d*2),
            nn.ReLU(),
            nn.Conv2d(self.d*2, self.d*2, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d2_d = nn.Sequential(
            nn.BatchNorm2d(self.d*2),
            nn.ReLU(),
            nn.Conv2d(self.d*2, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d_1 = nn.Sequential(
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, 3, kernel_size=(3,3), padding=1)
        )

        self.relu = nn.Sequential(nn.ReLU())
        self.c = nn.Parameter(torch.ones(1))

    def forward(self, image):
        image = image.detach()
        image = image.view(-1,3,32,32)
        image_rd = image.repeat((1,self.d//3,1,1))
        image_rd2 = image.repeat((1,2*self.d//3,1,1))
        
        x = self.res_block_1_d(image)
        x = x + image_rd

        x = self.res_block_d_d(x)
        x = x + image_rd

        x = self.res_block_d_d2(x)
        x = x + image_rd2
        
        x = self.res_block_d2_d2(x)
        x = x + image_rd2

        x = self.res_block_d2_d(x)
        x = x + image_rd

        x = self.res_block_d_1(x)
        #x = nn.functional.tanh(x)
        x = self.c * image + x
        
        return x

def unit_test():
    test_gen = Resnet_Editor(102)

    images = torch.rand((10, 3, 32, 32))              # Gets us 10 batches of noise (100 dim)
    output = test_gen.forward(images)
    print(output.shape)
    assert(images.shape == (10,3,32,32))
    utils.print_network(test_gen)

if __name__ == "__main__":
    unit_test()

