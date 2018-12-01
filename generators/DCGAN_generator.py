#!/usr/bin/python3
import torch
import torch.nn as nn
from utils import normal_init
import utils
        
class DCGAN_Generator(nn.Module):
    def __init__(self, d=256):
        super(small_DCGANGenerator, self).__init__()
        # ====== LAYERS ======
        self.d = d
        self.reshape_no_labels = nn.Sequential(
            nn.Linear(128, 4*4*self.d),
            nn.BatchNorm1d(4*4*self.d),
            nn.ReLU(True)
        )
        
        self.conv1 = nn.Sequential(
            # Conv1
            nn.ConvTranspose2d(self.d, self.d//2, kernel_size=2,stride=2),
            nn.BatchNorm2d(self.d//2),
            nn.ReLU(True),
            # Conv2
            nn.ConvTranspose2d(self.d//2, self.d//4, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.d//4),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            # Conv3 - We will not do a tanh here, it will be added later 
            nn.ConvTranspose2d(self.d//4, 3, kernel_size=2, stride=2),
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, noise):
        batch_n = noise.shape[0]
        x = self.reshape_no_labels(noise.view(batch_n, -1)).view(batch_n, self.d, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        
def unit_test():
    # Some sanity checks for the generator
    test_gen = DCGANGenerator(d=256)

    noise = torch.rand((10, 128))              # Gets us 10 batches of noise (100 dim)
    output_withlabels = test_gen.forward(noise)
    print(output_withlabels.shape) 
    assert(output_withlabels.shape == (10,3,32,32))
    
    utils.print_network(test_gen)

if __name__ == "__main__":
    unit_test()

