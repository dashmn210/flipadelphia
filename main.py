import yaml
import os
import argparse
from collections import namedtuple

import random
import numpy as np
import tensorflow as tf

from src.data.dataset import Dataset


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

    # use config to preprocess data
    d = Dataset(config)
    # if inference
        # restore all models in the spec from working_dir
    # elif training
        # train all models in spec

    # for each model in spec
        # inference on dataset

        # evaluate
            # categorical-specific
                # AUC
                # ROC curve
                # average feature correlation with this (cramer's V) 
            # continuous-specific
                # accuracy
                # MSE
                # residual plot
                # average feature correlation (point-biserial)
            # model-specific
                # conditional/marginal R^2
                # loss
                # attention dump? 
                # etc etc

    # clean up temp files?
#   cleanup(d)



# use args to preprocess data
    # possibly cut by column into sperate files
    # check vocab
    # get vocab size
    # possibly rm out-of-vocab tokens


