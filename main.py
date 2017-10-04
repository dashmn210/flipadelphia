import yaml
import os
import argparse
from collections import namedtuple

import random
import numpy as np
import tensorflow as tf

from src.data.dataset import Dataset
import src.msc.constants as constants

import src.models.dummies.tf_dummy as tf_dummy


def process_command_line():
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument('--config', dest='config', type=str, default='config.yaml', 
                        help='config file for this experiment')
    parser.add_argument('--inference', dest='inference', action='store_true', 
                        help='run inference')
    parser.add_argument('--train', dest='train', action='store_true', 
                        help='run training')
    parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='gpu')
    args = parser.parse_args()
    return args

def load_config(filename):
    d = yaml.load(open(filename).read())
    d = namedtuple("config", d.keys())(**d)
    return d

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)  # only default graph


if __name__ == '__main__':
    # parse args
    args = process_command_line()
    config = load_config(args.config)   

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(config.seed)
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    # use config to preprocess data
    d = Dataset(config)

    if args.train:
        for model_description in config.model_spec:
            model_dir = os.path.join(config.working_dir, model_description['type'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'],
                model_builder_class=tf_dummy.TFDummy)
            model.train(d, model_dir)
            model.save(model_dir)

    if args.inference:
        for model_description in config.model_spec:
            # TODO -- instantiate model
            model_dir = os.path.join(config.working_dir, model_description['type'])
            model.load(model_dir)
            predictions = model.inference(d, model_dir, dev=False)

            # TODO evaluate 
                # categorical-specific
                    # AUC
                    # ROC curve
                    # average feature correlation with this (cramer's V) 
                # continuous-specific
                    # accuracy
                    # MSE
                    # residual plot
                    # average feature correlation (point-biserial)
                # model-specific ( model.report() )
                    # conditional/marginal R^2
                    # loss
                    # attention dump? 
                    # etc etc

    # TODO
    utils.cleanup(d.data_by_variable)  # pointers to all cut files
    utils.cleanup(predictions)  # pointers to all predictions files
