import torch

import modules, util

@util.unittest
class Editor(modules.Savable, modules.NormalInit):

    def __init__(self, channels, width=64):
        super(Editor, self).__init__()
        
        self.cnn = modules.ResNet(
                
            modules.ResBlock(
                conv = torch.nn.Sequential(
                    
                    torch.nn.Conv2d(channels, width, 3, padding=1),
                    torch.nn.BatchNorm2d(width),
                    torch.nn.ReLU(),
                    
                    torch.nn.Conv2d(width, width, 3, padding=1),
                    torch.nn.BatchNorm2d(width)
                ),
                shortcut = torch.nn.Conv2d(channels, width, 1),
                activation = torch.nn.ReLU()
            ),
            
            modules.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(width, width, 3, padding=1),
                    torch.nn.BatchNorm2d(width),
                    torch.nn.ReLU(),
                    
                    torch.nn.Conv2d(width, width, 3, padding=1),
                    torch.nn.BatchNorm2d(width)
                ),
                activation = torch.nn.ReLU()
            ),
            
            # 16 -> 32
            modules.ResBlock(
                conv = torch.nn.Sequential(
                    torch.nn.Conv2d(width, width, 3, padding=1),
                    torch.nn.BatchNorm2d(width),
                    torch.nn.ReLU(),
                    
                    torch.nn.Conv2d(width, channels, 3, padding=1),
                ),
                shortcut = torch.nn.Conv2d(width, channels, 1),
                activation = torch.nn.Sequential()
            ),
            
        ) # ResNet
        
        self.init_weights(self.cnn)
    
    def get_init_targets(self):
        return [torch.nn.Conv2d]
    
    def forward(self, X, Y):
        return self.cnn(X)
    
    def unittest():
        
        CHANNELS = 3
        NOISESIZE = 128
        CLASSES = 10
        EMBEDSIZE = 16
        BATCHSIZE = 2
        IMAGESIZE = (32, 32)
        
        g = Generator(CHANNELS, NOISESIZE, CLASSES, EMBEDSIZE)
        
        Z = torch.rand(BATCHSIZE, NOISESIZE)
        Y = torch.LongTensor(BATCHSIZE).random_(0, CLASSES)
        
        X = g(Z, Y)
        assert X.size() == (BATCHSIZE, CHANNELS, *IMAGESIZE)
