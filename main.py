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
import time
import sys

from src.data.dataset import Dataset
import src.msc.constants as constants
import src.msc.utils as utils
import src.analysis.evaluator as evaluator
import src.models.neural.tf_dummy as tf_dummy
import src.models.neural.tf_flipper as tf_flipper

def process_command_line():
    """ returns a 1-tuple of cli args
    """
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
    """ loads YAML config into a named tuple
    """
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

    reload(sys)
    sys.setdefaultencoding('utf8')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(config.seed)
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    # use config to preprocess data
    start = time.time()
    print 'MAIN: parsing dataset'
    d = Dataset(config)
    print 'MAIN: dataset done. took %.2fs' % (time.time() - start)

    # if train, switch the dataset to train, then
    #  train and save each model in the config spec
    if args.train:
        d.set_active_split(config.train_suffix)

        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue

            print 'MAIN: training ', model_description['name']
            start_time = time.time()
            model_dir = os.path.join(config.working_dir, model_description['name'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'])

            model.train(d, model_dir)
            model.save(model_dir)
            print 'MAIN: training %s done, time %.2fs' % (
                model_description['name'], time.time() - start_time)

    # if test, switch thh datset to test, 
    #  and run inference + evaluation for each model
    #  in the config spec
    if args.test:
        d.set_active_split(config.test_suffix)

        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue
            print 'MAIN: inference with ', model_description['name']
            start_time = time.time()

            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'])

            model_dir = os.path.join(config.working_dir, model_description['name'])
            model.load(d, model_dir)

            predictions = model.inference(d, model_dir)
            utils.pickle(
                predictions, os.path.join(model_dir, 'predictions'))

            evaluation = evaluator.evaluate(config, d, predictions, model_dir)
            utils.pickle(
                evaluation, os.path.join(model_dir, 'evaluation'))
            evaluator.write_summary(evaluation, model_dir)

            print 'MAIN: evaluation %s done, time %.2fs' % (
                model_description['name'], time.time() - start_time)

    # TODO maybe some kind of cleanup of temporrary files? like
    # datasets, etc etc
    utils.cleanup()

