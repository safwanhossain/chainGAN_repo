# Import upper directories:

import sys
sys.path.append("..")

# Imports:

import torch
import torch.nn as nn
from utils import normal_init
import utils

# Class:

class resnet_discriminator(nn.Module):
    def __init__(self, edit_num, d):
        super(resnet_discriminator, self).__init__()
        self.d = d
        self.features = edit_num + 1
        self.multiLastLayer = nn.ModuleList([])
        # ====== LAYERS ======
        self.optRes = nn.Sequential(
            nn.Conv2d(3, self.d, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.AvgPool2d(2)
        )
        self.shortCut1 = nn.Sequential(
            nn.Conv2d(3, self.d, kernel_size = 1),
            nn.AvgPool2d(2)
        )
        self.resBlock1 = nn.Sequential(
            #nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.AvgPool2d(2)
        )
        self.shortCut2 = nn.Sequential(
            nn.Conv2d(self.d, self.d, kernel_size = 1),
            nn.AvgPool2d(2)
        )
        self.resBlock2 = nn.Sequential(
            #nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
        )
        self.shortCut3 = nn.Sequential(
            nn.Conv2d(self.d, self.d, kernel_size = 1),
        )
        self.resBlock3 = nn.Sequential(
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(self.d, self.d, kernel_size = (3, 3), padding = 1),
        )
        self.shortCut4 = nn.Sequential(
            nn.Conv2d(self.d, self.d, kernel_size = 1),
        )
        self.pool = nn.AvgPool2d(8)
        for i in range(self.features):
            lin = nn.Linear(self.d, 1)
            self.multiLastLayer.append(lin)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, x, numEditor, labels = None):
        batch_n = x.shape[0]
        x = torch.nn.functional.relu(self.optRes(x) + self.shortCut1(x))
        x = torch.nn.functional.relu(self.resBlock1(x) + self.shortCut2(x))
        x = torch.nn.functional.relu(self.resBlock2(x) + self.shortCut3(x))
        x = torch.nn.functional.relu(self.resBlock3(x) + self.shortCut4(x))
        out = self.pool(x).view(batch_n, self.d)
        out = self.multiLastLayer[numEditor](out).view(batch_n, 1)
        return out
        
def unit_test():
    # Some sanity checks for the generator
    test_dis = Resnet_Discriminator(3, 128)

    image = torch.rand((10,3,32,32))
    image = image.view(-1,3,32,32)

    output = test_dis.forward(image, 2)
    print(output.shape)
    assert(output.shape == (10,1))
    utils.print_network(test_dis)
    
if __name__ == "__main__":
    unit_test()
