import torch

import util

def index(t, i):
    assert len(t.size()) >= len(i.size())
    assert t.shape[:len(i.shape)] == i.shape
    compressed_i = compress_index(i, len(i.size()))
    squash = torch.numel(i)
    return t.view(squash, -1)[compressed_i].view(t.size())

def compress_index(i, C):
    if C == 1:
        return i
    else:
        return _compress_index(i, C)

def _compress_index(i, C):
    N, D = i.shape[-2:]
    add = torch.arange(N).long().to(i.device) * D
    shp = [1] * C
    shp[-2] = N
    add = add.view(shp)
    out = list(i.shape[:-1])
    out[-1] = N*D
    i = (i + add).view(out)
    return compress_index(i, C-1)

@util.unittest
class Unittest:

    def unittest():
        i = torch.LongTensor([
            [0, 1, 2, 3, 4]
        ] * 2)
        
        assert compress_index(i, 2).tolist() == list(range(10))
        
        i = torch.LongTensor([
            [0, 3, 2, 4, 1],
            [4, 2, 3, 1, 0]
        ])
        
        target = [0, 3, 2, 4, 1, 9, 7, 8, 6, 5]
        assert compress_index(i, 2).tolist() == target
        assert compress_index(i.unsqueeze(1), 3).tolist() == target
        
        v = torch.rand(4, 5, 2, 3)
        s, a = v.sort(-1)
        assert v.tolist() != s.tolist()
        w = index(v, a)
        assert w.tolist() == s.tolist()
