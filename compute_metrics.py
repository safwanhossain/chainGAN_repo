#!/usr/bin/python3

import argparse 
import json
import importlib
import sys
import torch
import numpy as np

from inception_score import inception_score

sys.path.append('./trainers')
from edit_and_gen import EditAndGen

def get_inception_score(generator, num_editors):
    # each batch is 32 images. We need 1570 batches
    all_images_for_edits = {}
    for i in range(1570):
        z = torch.randn(32,128)
        z = z.float().cuda()
        
        gen_images = []
        for images in G.forward(z):
            gen_images.append(torch.nn.functional.tanh(images))
        
        for j in range(num_editors+1):
            if j not in all_images_for_edits:
                all_images_for_edits[j] = []
            all_images_for_edits[j].append(gen_images[j].cpu().data.numpy())
    
    print("Generated all images")

    for j in range(num_editors+1):
        print("Computing Inception score for Editor: ", j)
        images = all_images_for_edits[j]
        images = np.concatenate(images, axis=0)
        images = images.reshape((-1, 3, 32, 32))
        mean, var = inception_score(list(images), resize=True)    
        print("Mean is: ", mean)
        print("Var is: ", var)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("param_json", help="json file pointing to hyper-param dict")
    parser.add_argument("save_dir", help="directory where models are saved")
    parser.add_argument("epoch", help="Epoch for which to compute inception/fid scores")

    config = parser.parse_args()
    with open(config.param_json, 'r') as fp:
        hyper_param_dict = json.load(fp)
    num_editors = hyper_param_dict["num_editors"]

    editor_name = hyper_param_dict['editor_name']
    editor_file = importlib.import_module("editors." + editor_name)
    editor_mod = getattr(editor_file, editor_name)
    
    generator_name = hyper_param_dict['generator_name']
    generator_file = importlib.import_module("generators." + generator_name)
    generator_mod = getattr(generator_file, generator_name)

    generator = generator_mod(d=hyper_param_dict['gen_dim'])
    editor_list = [editor_mod(d=hyper_param_dict['edit_dim']) \
            for i in range(hyper_param_dict["num_editors"])]
    G = EditAndGen(generator, editor_list, num_editors).cuda()
   
    LOAD_DIR = config.save_dir + '/model_trained_' + config.epoch + '.tar'
    save_dict = torch.load(LOAD_DIR)
    G.load_state_dict(save_dict['Generators'])
   
    print("Computing Inception scores for", config.save_dir, "on epoch", config.epoch)
    get_inception_score(G, num_editors) 

