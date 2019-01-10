import torch, random

import modules, util

@util.unittest
class Discriminator(modules.Savable, modules.NormalInit):

    def __init__(self, channels, imagesize, classes, embedsize, ed_n):
        super(Discriminator, self).__init__()
        
        assert imagesize == (32, 32)
        
        insize = channels + embedsize
        
        self.cnn = torch.nn.Sequential(
        
            modules.ResNet(
        
                # 32 -> 16
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(insize, 128, 3, padding=1),
                        torch.nn.ReLU(),
                        
                        torch.nn.Conv2d(128, 128, 3, padding=1, stride=2),
                        torch.nn.ReLU(),
                        
                        torch.nn.AvgPool2d(2),
                    ),
                    shortcut = torch.nn.Sequential(
                        torch.nn.Conv2d(insize, 128, 1),
                        torch.nn.AvgPool2d(2),
                    ),
                    activation = torch.nn.ReLU()
                ),
            
                # 16 -> 8
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(128, 128, 3, padding=1),
                        torch.nn.ReLU(),
                        
                        torch.nn.Conv2d(128, 128, 3, padding=1),
                        torch.nn.ReLU(),
                        
                        torch.nn.AvgPool2d(2),
                    ),
                    shortcut = torch.nn.AvgPool2d(2),
                    activation = torch.nn.ReLU()
                ),
                
                # 8 -> 8
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(128, 128, 3, padding=1),
                        torch.nn.ReLU(),
                        
                        torch.nn.Conv2d(128, 128, 3, padding=1),
                        torch.nn.ReLU()
                    ),
                    activation = torch.nn.ReLU()
                ),
                
                # 8 -> 8
                modules.ResBlock(
                    conv = torch.nn.Sequential(
                        torch.nn.Conv2d(128, 128, 3, padding=1),
                        torch.nn.ReLU(),
                        
                        torch.nn.Conv2d(128, 128, 3, padding=1),
                        torch.nn.ReLU()
                    ),
                    activation = torch.nn.ReLU()
                ),
            ), # ResNet

            torch.nn.AvgPool2d(8),
            modules.Reshape(128)
        ) # Sequential
        
        self.net = torch.nn.ModuleList([
            torch.nn.Linear(128, 1) for i in range(ed_n+1)
        ])
        
        self.emb = torch.nn.Embedding(classes, embedsize)
        
        self.init_weights(self.cnn)
        self.init_weights(self.net)
        self.init_weights(self.emb)
    
    def get_init_targets(self):
        return [torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding]
    
    def forward(self, X, Y, ed_i):
        emb = self.emb(Y)
        N, D = emb.size()
        emb = emb.view(N, D, 1, 1)
        N, C, W, H = X.size()
        emb = emb.repeat(1, 1, W, H)
        X = torch.cat([X, emb], dim=1)
        return self.net[ed_i](self.cnn(X))
    
    def unittest():
    
        CHANNELS = 3
        IMAGESIZE = (32, 32)
        CLASSES = 10
        EMBEDSIZE = 16
        BATCHSIZE = 64
        
        d = Discriminator(CHANNELS, IMAGESIZE, CLASSES, EMBEDSIZE)
        img = torch.rand(BATCHSIZE, CHANNELS, *IMAGESIZE)
        lab = torch.LongTensor(BATCHSIZE).random_(0, 10)
        cnn = d(img, lab)
        assert cnn.size() == (BATCHSIZE, 1)
