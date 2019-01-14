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

class resnet_editor(nn.Module):
    def __init__(self, d):
        super(resnet_editor, self).__init__()
        self.d = d
        assert self.d > 0, "Resnet dimension must be greater than zero. Sorry!"
        assert self.d % 3 == 0, "Resnet dimension must be divisible by 3. Sorry!"
        
        self.res_block_1_d = nn.Sequential(
            nn.Conv2d(3, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1)
        )
        self.sc_1_d = torch.nn.Conv2d(3, self.d, 1)

        self.res_block_d_d = nn.Sequential(
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1)
        )

        self.res_block_d_d2 = nn.Sequential(
            nn.Conv2d(self.d, self.d*2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d*2),
            nn.ReLU(),
            nn.Conv2d(self.d*2, self.d*2, kernel_size=(3,3), padding=1)
        )
        self.sc_d_d2 = torch.nn.Conv2d(self.d, self.d*2, 1)
        
        self.res_block_d2_d2 = nn.Sequential(
            nn.Conv2d(self.d*2, self.d*2, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d*2),
            nn.ReLU(),
            nn.Conv2d(self.d*2, self.d*2, kernel_size=(3,3), padding=1)
        )
        
        self.res_block_d2_d = nn.Sequential(
            nn.Conv2d(self.d*2, self.d, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(self.d),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size=(3,3), padding=1)
        )
        self.sc_d2_d = torch.nn.Conv2d(self.d*2, self.d, 1)
        
        self.res_block_d_1 = nn.Conv2d(self.d, 3, kernel_size=(3,3), padding=1)

    def forward(self, image):
        x = image.detach()
        x = torch.nn.functional.relu(self.res_block_1_d(x) + self.sc_1_d(x))
        x = torch.nn.functional.relu(self.res_block_d_d(x) + x)
        x = torch.nn.functional.relu(self.res_block_d_d2(x) + self.sc_d_d2(x))
        x = torch.nn.functional.relu(self.res_block_d2_d2(x) + x)
        x = torch.nn.functional.relu(self.res_block_d2_d(x) + self.sc_d2_d(x))
        return self.res_block_d_1(x)

def unit_test():
    test_gen = resnet_editor(102)

    images = torch.rand((10, 3, 32, 32))              # Gets us 10 batches of noise (100 dim)
    output = test_gen.forward(images)
    print(output.shape)
    assert(images.shape == (10,3,32,32))
    utils.print_network(test_gen)

if __name__ == "__main__":
    unit_test()

