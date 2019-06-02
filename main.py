#!/usr/bin/python3

import argparse 
import json
import importlib

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("param_json", help="json file pointing to hyper-param dict")
    parser.add_argument("save_dir", help="directory to save the images")
    config = parser.parse_args()
    with open(config.param_json, 'r') as fp:
        hyper_param_dict = json.load(fp)
 
    editor_name = hyper_param_dict['editor_name']
    generator_name = hyper_param_dict['generator_name']
    discriminator_name = hyper_param_dict['discriminator_name']
    trainer_name = hyper_param_dict['trainer_name']

    editor_file = importlib.import_module("editors." + editor_name)
    generator_file = importlib.import_module("generators." + generator_name)
    discriminator_file = importlib.import_module("discriminators." + discriminator_name)
    trainer_file = importlib.import_module("trainers." + trainer_name)

    editor_mod = getattr(editor_file, editor_name)
    generator_mod = getattr(generator_file, generator_name)
    discriminator_mod = getattr(discriminator_file, discriminator_name)
    trainer_mod = getattr(trainer_file, trainer_name)

    if not hyper_param_dict['is_biggan']:
        generator = generator_mod(d=hyper_param_dict['gen_dim'])
        editor_list = [editor_mod(d=hyper_param_dict['edit_dim']) \
                for i in range(hyper_param_dict["num_editors"])]
        discriminator = discriminator_mod(edit_num=hyper_param_dict["num_editors"], d=hyper_param_dict['dis_dim'])
    else:
        generator = generator_mod(hyper_param_dict['gen_list'],hyper_param_dict['pic_dim'],hyper_param_dict['z_dim'],hyper_param_dict['rgb'])
        editor_list = [editor_mod(hyper_param_dict['editor_list'],hyper_param_dict['rgb']) \
                for i in range(hyper_param_dict["num_editors"])]
        discriminator = discriminator_mod(hyper_param_dict['dis_list'],hyper_param_dict["num_editors"],hyper_param_dict['pic_dim'],hyper_param_dict['rgb'])
    train_data, train_labels = utils.get_data()
    my_dataset = utils.create_dataset(train_data, train_labels, int(hyper_param_dict['batch_size'])) 
    GAN = trainer_mod(my_dataset, generator, discriminator, hyper_param_dict, config.save_dir, \
            editor_object_list=editor_list, num_editors=hyper_param_dict["num_editors"])
    
    GAN.train()


