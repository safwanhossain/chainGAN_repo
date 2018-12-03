#!/usr/bin/python3
# Imports:

import torch
import torch.nn as nn
import sys

sys.path.append('../')
from utils import normal_init

class EditAndGen(nn.Module):
    def __init__(self, base_gen_object, editor_object_list, num_edit):
        super(EditAndGen, self).__init__()
        
        # @yuchen commenting these lines out saves no significant memory.
        # these modules are truly tiny.
        
        self.mods = nn.ModuleList([base_gen_object])
        self.mods.extend(editor_object_list)
        self.param_groups = [list(m.parameters()) for m in self.mods]

    def weight_init(self, mean, std):
        for m in self.mods:
            normal_init(m, mean, std)
    
    def forward(self, z):
        curr_inp = z
        all_images = []
        for i in range(len(self.mods)):
            curr_inp = self.mods[i](curr_inp).detach()
            all_images.append(curr_inp.clone())
        return all_images
        
    def generate_upto(self, z, upto):
        """ Generates upto but not including the var upto"""
        curr_inp = z
        i = -1
        for i in range(upto - 1):
            curr_inp = self.mods[i](curr_inp).clone().detach()
        curr_inp = self.mods[i+1](curr_inp.clone())
        return curr_inp
        
class EditAndGenLabels(nn.Module):
    
    def __init__(self, base_gen_class, base_edit_class, num_edit):
        super(EditAndGenLabels, self).__init__()
        
        base_gen = base_gen_class()
        self.add_module('base', base_gen)
        self.param_groups = [list(base_gen.parameters())]
        for i in range(num_edit):
            base_edit = base_edit_class()
            self.add_module('edit%d'%i, base_edit)
            self.param_groups.append(list(base_edit.parameters()))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, x, labels):
        generated = []
        for layer in self.named_children():
            if len(generated) == 0:
                image = layer[1](x, labels)
                generated.append(image)
            else:
                image = layer[1](generated[-1], labels)
                generated.append(image)
        return generated

def test():
    from generator import TinyGenerator
    from edit_generator import edit_generator_tiny
    from utils import print_network
    test_edit_gen = EditAndGen(TinyGenerator, edit_generator_tiny, 50)
    
    assert(len(test_edit_gen.param_groups) == 51)
    
    z = torch.randn((10, 64))
    gen = test_edit_gen(z)
    
    for i, pic in enumerate(gen):
        print('At step %d'%i, pic.shape)
    
    print_network(test_edit_gen)
    
if __name__ == '__main__':
    test()
