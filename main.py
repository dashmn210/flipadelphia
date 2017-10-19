"""
TODO -- automatically divvy up data for continuous
         confounds and mixed regression?

"""
import yaml
import os
import argparse
from collections import namedtuple
import pickle
import random
import numpy as np
import tensorflow as tf

from src.data.dataset import Dataset
import src.msc.constants as constants
import src.msc.utils as utils
import src.analysis.evaluator as evaluator
import src.models.neural.tf_dummy as tf_dummy
import src.models.neural.tf_flipper as tf_flipper

def process_command_line():
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument('--config', dest='config', type=str, default='config.yaml', 
                        help='config file for this experiment')
    parser.add_argument('--test', dest='test', action='store_true', 
                        help='run test')
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


def validate_config(config):
    # TODO
    return True



if __name__ == '__main__':
    # parse args
    args = process_command_line()
    config = load_config(args.config)   
    validate_config(config)


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(config.seed)
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    # use config to preprocess data
    d = Dataset(config)

    if args.train:
        d.set_active_split(config.train_suffix)

        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue

            print 'MAIN: training ', model_description['type']
            model_dir = os.path.join(config.working_dir, model_description['type'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'])

            model.train(d, model_dir)
            model.save(model_dir)

    if args.test:
        d.set_active_split(config.test_suffix)

        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue
            print 'MAIN: inference with ', model_description['type']

            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'])

            model_dir = os.path.join(config.working_dir, model_description['type'])
            model.load(d, model_dir)

            predictions = model.inference(d, model_dir)
            utils.pickle(
                predictions, os.path.join(model_dir, 'predictions'))

            evaluation = evaluator.evaluate(config, d, predictions, model_dir)
            utils.pickle(
                evaluation, os.path.join(model_dir, 'evaluation'))


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
