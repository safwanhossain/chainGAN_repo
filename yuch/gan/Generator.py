import torch

import modules, util

@util.unittest
class Generator(modules.Savable, modules.NormalInit):

    def __init__(self, channels, noisesize, classes, embedsize, width=128):
        super(Generator, self).__init__()
        
        self.emb = torch.nn.Embedding(classes, embedsize)
        
        self.cnn = torch.nn.Sequential(
        
            torch.nn.Linear(noisesize + embedsize, width*4*4),
            
            modules.Reshape(width, 4, 4),
            
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(),
            
            modules.ResNet(
                
                # 4 -> 8
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        
                        torch.nn.Conv2d(width, width, 3, padding=1),
                        torch.nn.BatchNorm2d(width),
                        torch.nn.ReLU(),
                        
                        modules.Upsample(scale_factor=2),
                        
                        torch.nn.Conv2d(width, width, 3, padding=1),
                        torch.nn.BatchNorm2d(width)
                    ),
                    shortcut = modules.Upsample(scale_factor=2),
                    activation = torch.nn.ReLU()
                ),
                
                # 8 -> 16
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        
                        torch.nn.Conv2d(width, width, 3, padding=1),
                        torch.nn.BatchNorm2d(width),
                        torch.nn.ReLU(),
                        
                        modules.Upsample(scale_factor=2),
                        
                        torch.nn.Conv2d(width, width, 3, padding=1),
                        torch.nn.BatchNorm2d(width)
                    ),
                    shortcut = modules.Upsample(scale_factor=2),
                    activation = torch.nn.ReLU()
                ),
                
                # 16 -> 32
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        
                        torch.nn.Conv2d(width, width, 3, padding=1),
                        torch.nn.BatchNorm2d(width),
                        torch.nn.ReLU(),
                        
                        modules.Upsample(scale_factor=2),
                        
                        torch.nn.Conv2d(width, width, 3, padding=1),
                        torch.nn.BatchNorm2d(width)
                    ),
                    shortcut = modules.Upsample(scale_factor=2),
                    activation = torch.nn.ReLU()
                ),
                
            ), # ResNet
            
            torch.nn.Conv2d(width, channels, 3, padding=1)
            
        ) # Sequential
        
        self.init_weights(self.emb)
        self.init_weights(self.cnn)
    
    def get_init_targets(self):
        return [torch.nn.Embedding, torch.nn.Linear, torch.nn.Conv2d]
    
    def forward(self, Z, Y):
        emb = self.emb(Y)
        cnn = torch.cat([Z, emb], dim=1)
        return self.cnn(cnn)
    
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
