import torch, tqdm, os, shutil, math, random

import mailupdater

import scipy.misc

import constants

from .Discriminator import Discriminator
from .Generator import Generator
from .Editor import Editor
from .Chain import Chain

USERNAME = "ychnlgy@gmail.com"
MAIL_FREQ = 1

class GAN:

    def __init__(self,
        name,
        dataloader,
        validloader,
        channels, 
        imagesize,
        noisesize,
        classes,
        G_embedsize,
        D_embedsize,
        uselabels,
        batchsize,
        ncritic,
        numeditors = 5,
        pretrain = True,
        gradpenalty = 10,
        cycle = 10,
        sample = 20,
        epoch = 2000,
        device = "cuda"
    ):
        self.name = name
        self.D = Discriminator(channels, imagesize, classes, D_embedsize, numeditors).to(device)
        self.G = Chain(
            Generator(channels, noisesize, classes, G_embedsize, width=96),
            [Editor(channels, width=64) for i in range(numeditors)]
        ).to(device)
        
        self.pretrain = pretrain
        self.data = dataloader
        self.validationset = validloader
        self.uselabels = uselabels
        self.batchsize = batchsize
        self.device = device
        self.classes = classes
        self.noisesize = noisesize
        self.cycle = cycle
        self.epoch = epoch
        self.ncritic = ncritic
        self.gradpenalty = gradpenalty
        self.G_file = "G-%s.torch" % name
        self.D_file = "D-%s.torch" % name
        self.sample = sample
        self.numeditors = numeditors
        
        self.G_sample = constants.RESULTS + "-" + name
        G_sample = os.path.join(self.G_sample, ["%d-ed%d.png", "%d-ed%d (C%d).png"][uselabels])
        if uselabels:
            self.output = lambda i, ed, y: G_sample % (i, ed, y.item())
        else:
            self.output = lambda i, ed, y: G_sample % (i, ed)
        
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0, 0.9))
        self.G_optim = [
            torch.optim.Adam(self.G.get_ed(i).parameters(), lr=1e-4, betas=(0, 0.9))
            for i in range(numeditors+1)
        ]
        
        self.D_sched = torch.optim.lr_scheduler.StepLR(self.D_optim, epoch//3)
        self.G_sched = [
            torch.optim.lr_scheduler.StepLR(self.G_optim[i], epoch//3)
            for i in range(numeditors + 1)
        ]
        
        print("Discriminator parameters: %d" % self.D.paramcount())
        print("Generator     parameters: %d" % self.G.paramcount())
        
        self.msg = "[%s] %.2f R | %.2f F"
        
        self.service = mailupdater.Service(USERNAME)
        
        if self.uselabels:
            self.get_progress = self._get_progress
            self.msg += " | %.2f M"
    
    def train(self):
    
        if self.pretrain:
            for i, args, bar in self.iter_data(1):
                self.update_discriminator(*args[0])
                
                if not i % self.cycle:
                    bar.set_description(self.msg % ("Pretrain", *self.get_progress()))
    
        for epoch in range(1, self.epoch+1):
        
            self.D_sched.step()
            [self.G_sched[i].step() for i in range(self.numeditors+1)]
        
            for i, args, bar in self.iter_data(self.ncritic):
                for j in range(self.ncritic):
                    self.update_discriminator(*args[j])
                self.update_generator(*args[-1])
            
                if not i % self.cycle:
                    bar.set_description(self.msg % ("E%d" % epoch, *self.get_progress()))
            
            self.report_validation_divergence()
            
            if not (epoch-1) % MAIL_FREQ:
            
                self.D.save(self.D_file)
                self.G.save(self.G_file)
                fnames = self.sample_images()
            
                with self.service.create("%s - epoch %d" % (self.name, epoch)) as email:
                    list(map(email.attach, fnames))
                    email.write(self.report_validation_divergence())
    
    # === PRIVATE ===
    
    def report_validation_divergence(self):
        N = V = 0.0
        with torch.no_grad():
            for X, Y in self.validationset:
                X = X.to(self.device)
                Y = Y.to(self.device)
                T = random.randint(0, self.numeditors)
                V += self.D(X, Y, T).mean().item()
                N += 1.0
        V /= N
        T = self.R/self.N
        F = self.F/self.N
        R = abs(T-V)/abs(T-F)
        return "Validation score: %.2f and divergence: %.2f" % (V, R)
    
    def sample_images(self):
        return list(self._sample_images())
    
    def _sample_images(self):
    
        if os.path.isdir(self.G_sample):
            shutil.rmtree(self.G_sample)
        
        os.makedirs(self.G_sample)
    
        self.G.eval()
    
        with torch.no_grad():
            Z = self.create_noise(self.sample)
            Y = self.create_uniform_labels()
            Xh_all = self.G(Z, Y, ed_i=all)
            for ed_i, Xh in enumerate(Xh_all):
                Xh = Xh.cpu().detach().permute(0, 2, 3, 1).squeeze(-1).numpy()
                for i, (x, y) in enumerate(zip(Xh, Y)):
                    fname = self.output(i, ed_i, y)
                    scipy.misc.imsave(fname, x)
                    yield fname
    
    def create_uniform_labels(self):
        uniform = torch.arange(self.classes).long()
        repeat = self.sample//self.classes
        Y = uniform.unsqueeze(-1).repeat(1, repeat).view(-1)
        return Y.to(self.device)
    
    def iter_data(self, nbatch=1):
        it = self.infinite_data()
        n = math.ceil(len(self.data)/nbatch)
        bar = tqdm.tqdm(range(1, n+1), ncols=80)
        self.R = self.M = self.F = self.N = 0.0
        for i in bar:
            yield i, [next(it) for i in range(nbatch)], bar
    
    def infinite_data(self):
        while True:
            for e in self._iter_data():
                yield e
    
    def _iter_data(self):
        for X, Y in self.data:
            X = X.to(self.device)
            Y = Y.to(self.device)
            T = random.randint(0, self.numeditors)
            yield X, Y, T
    
    def update_discriminator(self, X, Y, T):
    
        self.D.train()
        self.G.eval()
    
        real = self.D(X, Y, T).mean()
        self.R += real.item()
        
        # Generated data
        
        Z = self.create_noise(len(X))
        with torch.no_grad():
            Xh = self.G(Z, Y, T).detach()
        
        fake = self.D(Xh, Y, T).mean()
        self.F += fake.item()
        D_loss = fake - real + self.grad_penalty(X, Xh, Y, T)
        
        # Mislabelled data
        
        if self.uselabels:
            pred = self.D(X, self.mislabel(Y), T).mean()
            self.M += pred.item()
            D_loss += pred
        
        self.D_optim.zero_grad()
        D_loss.backward()
        self.D_optim.step()
        
        self.N += 1

    def update_generator(self, X, Y, T):
        
        self.D.eval()
        self.G.train()
        
        Z = self.create_noise(len(Y))
        Xh = self.G(Z, Y, T)
        G_loss = -self.D(Xh, Y, T).mean()

        self.G_optim[T].zero_grad()
        G_loss.backward()
        self.G_optim[T].step()

    def _get_progress(self):
        return [self.R/self.N, self.F/self.N, self.M/self.N]
    
    def get_progress(self):
        return [self.R/self.N, self.F/self.N]
    
    def mislabel(self, Y):
        add = torch.LongTensor(Y.size()).random_(1, self.classes-1).to(self.device)
        out = (Y + add) % self.classes
        assert (Y != out).all()
        return out
    
    def create_noise(self, N=None):
        if N is None:
            N = self.batchsize
        Z = torch.FloatTensor(N, self.noisesize).normal_(mean=0, std=1)
        return Z.to(self.device)
    
    def grad_penalty(self, X, Xh, Y, T):
        N = len(X)
        V = X.device
        
        alpha = torch.FloatTensor(N, 1, 1, 1).uniform_(0, 1).to(V)
        intrp = alpha * Xh + (1 - alpha) * X
        intrp = torch.autograd.Variable(intrp, requires_grad=True)
        score = self.D(intrp, Y, T)
        goutp = torch.ones(score.size()).to(V)
        grads = torch.autograd.grad(
            outputs = score,
            inputs = intrp,
            grad_outputs = goutp,
            create_graph = True,
            retain_graph = True,
            only_inputs = True
        )[0].view(N, -1)
        
        value = (grads.norm(2, dim=1)-1)**2
        return value.mean() * self.gradpenalty
