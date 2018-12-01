#!/usr/bin/python3

import argparse 
import json
import importlib

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("generator", help="Name of generator class")
    parser.add_argument("discriminator", help="Name of discriminator class")
    parser.add_argument("editor", help="Name of editor class")
    parser.add_argument("trainer", help="Name of trainer class")
    parser.add_argument("num_editors", help="Name of editor class")
    parser.add_argument("param_json", help="json file pointing to hyper-param dict")
    parser.add_argument("save_dir", help="directory to save the images")
    config = parser.parse_args()

    with open(config.param_json, 'r') as fp:
        hyper_param_dict = json.load(fp)

    editor_name = "editors." + config.editor
    generator_name = "generators." + config.generator
    discriminator_name = "discriminators." + config.discriminator
    trainer_name = "trainers." + config.trainer

    editor_file = importlib.import_module(editor_name)
    editor_mod = getattr(editor_file, config.editor)
    generator_file = importlib.import_module(generator_name)
    generator_mod = getattr(generator_file, config.generator)
    discriminator_file = importlib.import_module(discriminator_name)
    discriminator_mod = getattr(discriminator_file, config.discriminator)
    trainer_file = importlib.import_module(trainer_name)
    trainer_mod = getattr(trainer_file, config.trainer)

    generator = generator_mod(d=hyper_param_dict['gen_dim'])
    editor_list = [editor_mod(d=hyper_param_dict['edit_dim']) for i in range(int(config.num_editors))]
    discriminator = discriminator_mod(edit_num = int(config.num_editors), d=hyper_param_dict['dis_dim'])
    
    train_data, train_labels = utils.get_data()
    my_dataset = utils.create_dataset(train_data, train_labels, int(hyper_param_dict['batch_size'])) 
    GAN = trainer_mod(my_dataset, generator, discriminator, hyper_param_dict, config.save_dir, \
            editor_object_list=editor_list, num_editors=int(config.num_editors))
    
    GAN.train()


