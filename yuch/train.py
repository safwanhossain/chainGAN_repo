#!/usr/bin/python3

import torch

import util, gan, datasets

@util.main
def main(
    dataset,
    uselabels,
    name,
    ncritic,
    datalimit = None,
    cycle=1,
    download=0,
    batch=100,
    noisesize=96,
    G_embedsize=32,
    D_embedsize=3,
):
    ncritic = int(ncritic)
    uselabels = int(uselabels)
    cycle = int(cycle)
    download = int(download)
    batch = int(batch)
    noisesize = int(noisesize)
    G_embedsize = int(G_embedsize)
    D_embedsize = int(D_embedsize)
    
    data = {
        "mnist": datasets.mnist,
        "cifar": datasets.cifar
    }[dataset]
    
    train_X, train_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE = data.get(download)
    
    if datalimit is not None:
        datalimit = int(datalimit)
        train_X = train_X[:datalimit].repeat(len(train_X)//datalimit, 1, 1, 1)
    
    dataloaders = []
    
    if not uselabels:
        CLASSES = 1
        noisesize += G_embedsize
        G_embedsize = 1
        D_embedsize = 1
    
    for X, Y in [
        (train_X, train_Y),
        (test_X, test_Y)
    ]:
        if not uselabels:
            Y = None
        loader = datasets.util.create_loader(batch, X, Y)
        dataloaders.append(loader)
    
    net = gan.GAN(
        name,
        dataloaders[0],
        dataloaders[1],
        CHANNELS,
        IMAGESIZE,
        noisesize,
        CLASSES,
        G_embedsize,
        D_embedsize,
        uselabels,
        batch,
        ncritic,
        cycle = cycle,
    )
    
    net.train()
