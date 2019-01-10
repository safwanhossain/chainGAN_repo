import torch

import modules

class Chain(modules.Savable):

    def __init__(self, generator, editors):
        super(Chain, self).__init__()
        self.ed = torch.nn.ModuleList([generator] + editors)
    
    def get_ed(self, i):
        return self.ed[i]
    
    def forward(self, Z, Y, ed_i):
        if ed_i is all:
            return self.gather_allresults(Z, Y)
        else:
            return self.gather_oneresult(Z, Y, ed_i)
    
    def prep(self, X):
        return torch.tanh(X)*0.5 + 0.5
    
    def gather_oneresult(self, Z, Y, ed_i):
        X = Z
        with torch.no_grad():
            for i in range(ed_i):
                X = self.ed[i](X).detach()
        return self.prep(self.ed[ed_i](X))
    
    def gather_allresults(self, Z, Y):
        out = []
        X = Z
        for ed in self.ed:
            X = ed(X)
            out.append(self.prep(X))
        return out
