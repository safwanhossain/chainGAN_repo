#!/usr/bin/python3

import time, torch, numpy, collections
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, random, sys
from trainers.edit_and_gen import EditAndGen

sys.path.append("../")
import utils

def generate_images(generator, directory_name, is_gpu=True, w_labels=False, epoch='na'):
    num_samples = 8

    z = torch.randn((num_samples, 128))

    labels = numpy.random.choice(10,num_samples)
    onehot_labels = numpy.eye(10)[labels]
    onehot_labels = torch.from_numpy(onehot_labels)
    onehot_labels = onehot_labels.view(-1,1,1,10)

    if is_gpu:
        z, onehot_labels = z.cuda(), onehot_labels.cuda()

    if w_labels:
        sample = generator(z, onehot_labels)
    else:
        sample = generator(z)
            
    for j in range(num_samples):
        for i, image in enumerate(sample):
            image = nn.functional.tanh(image)
            image = image[j].view(3,32,32).mul(0.5).add(0.5)
            filename = "def"
            if w_labels:
                filename = "Ep" + epoch + "_" + "sample%d"%j + "_edited%d"%i + "_label: " + str(labels[i])
            else:
                filename = "Ep" + epoch + "_" + "sample%d"%j + "_edited%d"%i 
                
            utils.save_image(image.cpu().detach().numpy(), filename, directory_name)
    
class base_chain_gan(object):
    def __init__(self, data_loader, generator, discriminator, hyper_param_dict, directory_name, editor_object_list=None, num_editors=0):
        # parameters
        self.epoch = hyper_param_dict['epoch']
        self.batch_size = hyper_param_dict['batch_size']
        self.pretrain_iter = hyper_param_dict['pretrain_iter']
        self.data_loader = data_loader
        self.gpu_mode = True

        self.num_edit_generators = num_editors   # number of generator we want
        self.gp = hyper_param_dict['gp']
        self.ncritic = hyper_param_dict['ncritic']
        self.lr = hyper_param_dict['lr']
        self.betas = hyper_param_dict['betas']
        self.directory_name = directory_name

        # Variables holding statistics of training
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['E_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.data_sampler = self.get_next_sample()

        # Generators
        self.G = EditAndGen(generator, editor_object_list, self.num_edit_generators)
        self.G.weight_init(mean=0.0, std=0.02)
        self.G_optimizers = []
        for optimizer in range(self.num_edit_generators + 1):
            self.G_optimizers.append(optim.Adam(self.G.param_groups[optimizer], lr=self.lr, betas=self.betas))
        if self.gpu_mode:
            self.G.cuda()
        
        # Discriminators
        self.D = discriminator
        self.D.weight_init(mean=0.0, std=0.02)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=self.betas)
        if self.gpu_mode:
            self.D.cuda()
        
        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')
    
    def get_next_sample(self):
        while True:
            gen = iter(self.data_loader)
            for images, labels in gen:
                yield images, labels
    
    def train(self):
        # Create the directory to save the generated images
        if not os.path.exists(self.directory_name):
            os.makedirs(self.directory_name)
        
        def process_variables(var_list):
            new_vars = []
            for var in var_list:
                if self.gpu_mode:
                    var = var.cuda()
                new_vars.append(var)
            return new_vars

        def backDifferentiate(loss, optimizer):
            return backDifferentiateMany(loss, [optimizer])
        
        def backDifferentiateMany(loss, optimizers):
            for optim in optimizers:
                optim.zero_grad()
            lossval = loss.item()
            loss.backward(retain_graph=True)
            for optim in optimizers:
                optim.step()
            return lossval
        
        def train_discriminator(update_index):
            self.D.train()
            self.G.eval()
           
            for _ in range(self.ncritic):
                z = torch.randn((self.batch_size, 128))
                images, labels = next(self.data_sampler)
                images = images.view(-1,3,32,32)
                images = images.cuda()
                z = z.cuda()
                real_loss = -torch.mean(self.D(images, update_index))

                # In training the discrim, we don't want to store the generator graphs but still be on gpu
                generated = self.G.generate_upto(z, update_index+1)     # generates upto but not including the editor number
                generated = nn.functional.tanh(generated)
                generated = generated.detach()
                generated = generated.cuda()
                
                fake_loss = torch.mean(self.D(generated, update_index))
                grad_penalty = utils.grad_penalty(images, generated, self.gp, self.D, update_index)
                loss = real_loss + fake_loss + grad_penalty
                backDifferentiate(loss, self.D_optimizer)

            self.train_hist['D_loss'].append(loss.item())
            return images 

        def train_generators(update_index):
            self.G.train()
            self.D.eval()
            
            z = torch.randn((self.batch_size, 128))
            z = process_variables([z])[0]
            
            gen_image = self.G.generate_upto(z, update_index+1)
            gen_image = nn.functional.tanh(gen_image)
            loss = -self.D(gen_image, update_index).mean()
            genLoss = backDifferentiate(loss, self.G_optimizers[update_index])

        def pre_train(pretraining_epoch=self.pretrain_iter):
            for epoch in range(pretraining_epoch): 
                num_iters = len(self.data_loader)
                for i in tqdm(range(num_iters), total=num_iters): 
                    update_index = random.randint(0, self.num_edit_generators) 
                    train_discriminator(update_index)
                        
        # In each epoch, run thru all the samples in data_loader
        print("Start pre-train")
        pre_train()
        print("End pre-train")

        for epoch in range(self.epoch):
            num_iters = len(self.data_loader)
            for i in tqdm(range(num_iters), total=num_iters): 
                update_index = random.randint(0, self.num_edit_generators) 
                sample_images = train_discriminator(update_index)
                train_generators(update_index)

            print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f" %
                          ((epoch + 1), (i + 1), len(self.data_loader), self.train_hist['D_loss'][-1]))
            
            generate_images(self.G, self.directory_name, w_labels=False, epoch=str(epoch))
            #utils.plot_editor_scores(self.G, self.D, self.gpu_mode, self.num_edit_generators, 
            #        directory_name + "/d_scores", epoch) 
            #utils.compute_wass_distance(sample_images, self.D, self.G, directory_name + "/wass_distance", epoch)
            
            if (epoch % 25)==0 and epoch != 0:
                G_optimizers_dict = [g_optim.state_dict() for g_optim in self.G_optimizers]
                save_dict = {'epoch' : epoch,
                             'Discriminator' : self.D.state_dict(),
                             'Generators' : self.G.state_dict(),
                             'D_optimizers' : G_optimizers_dict }
                file_name = directory_name + '/model_trained_' + str(epoch) + '.tar'
                torch.save(save_dict, file_name)
            
def train():
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if not os.path.exists(directory_name + '/d_scores'):
        os.makedirs(directory_name + '/d_scores')
    if not os.path.exists(directory_name + '/wass_distance'):
        os.makedirs(directory_name + '/wass_distance')
    
    mnist_train_data, mnist_train_labels = utils.get_data()
    
    my_dataset = utils.create_dataset(mnist_train_data, mnist_train_labels, 64) 
    test_gan = base_chain_gan(my_dataset)
    
    test_gan.train()

if __name__ == "__main__":
    train() 

