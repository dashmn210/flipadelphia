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
import copy
from collections import defaultdict
import pandas as pd

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)  # only default graph


def run_experiment(config, args):
    # if train, switch the dataset to train, then
    #  train and save each model in the config spec
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)
    utils.write_config(config, os.path.join(config.working_dir, 'config.yaml'))

    print 'MAIN: parsing dataset'
    d = Dataset(config)
    print 'MAIN: dataset done. took %.2fs' % (time.time() - start)

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
        results = defaultdict(list)  # items to be written in executive summary 
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
            # store info for executive summary
            results['model-name'].append(model_description['name'])
            results['params'].append(str(model_description['params']))
            results['correlation'].append(evaluation['mu_corr'])
            results['performance'].append(evaluation['mu_perf'])
            results['model-dir'].append(model_dir)

            print 'MAIN: evaluation %s done, time %.2fs' % (
                model_description['name'], time.time() - start_time)

        return results

def validate_config(config):
    num_expts = config.num_experiments

    model_params = [
        x for m in config.model_spec \
        for x in (m['params'] or {}).values()
    ]
    if not any(isinstance(x, list) for x in model_params):
        if num_expts != 1:
            print 'MAIN: falling back to 1 experiment...no randomizable values provided'
        return 1
    return num_expts


def generate_experiment(parent_config, expt_id):
    """ choose a random thing from each config list 
    """
    d = copy.deepcopy(dict(parent_config._asdict()))

    assert not d['working_dir'].endswith('/')
    d['working_dir'] = d['working_dir'] + '_%s' % expt_id

    for model_config in d['model_spec']:
        if not model_config['params']: 
            continue
        for k, v in model_config['params'].items():
            if isinstance(v, list):
                model_config['params'][k] = random.choice(v)
    return namedtuple("config", d.keys())(**d)


if __name__ == '__main__':
    # parse args
    args = process_command_line()
    config = utils.load_config(args.config)   
    num_experiments = validate_config(config)
    reload(sys)
    sys.setdefaultencoding('utf8')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(config.seed)
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    # use config to preprocess data
    start = time.time()

    try:
        results = None
        for i in range(num_experiments):
            expt = generate_experiment(config, i)
            if os.path.exists(os.path.join(expt.working_dir, 'config.yaml')):
                print 'MAIN: skipping expt ', i
                continue
            result = run_experiment(expt, args)
            if results is None:
                results = result
            else:
                for k in results:
                    results[k] = results[k] + result[k]
    except KeyboardInterrupt:
        pass
    finally:
        executive_summary_df = pd.DataFrame.from_dict(results)

    # now write the summary to a csv at the parent's working dir
    summary_path = os.path.join(config.working_dir, 'summary.csv')
    if os.path.exists(summary_path):
        with open(summary_path, 'a') as f:
            executive_summary_df.to_csv(summary_path, header=False)
    else:
        executive_summary_df.to_csv(summary_path)
    print 'MAIN: wrote summary to ', summary_path

    # TODO maybe some kind of cleanup of temporrary files? like
    # datasets, etc etc
    utils.cleanup()

