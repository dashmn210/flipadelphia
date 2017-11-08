"""
TODO -- automatically divvy up data for continuous
         confounds and mixed regression?

"""
import yaml
import os
import csv
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
import traceback

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
    parser.add_argument('--model', dest='model', type=str, default=None,
                        help='force a single model to be run (turns off multiple expts)')
    parser.add_argument('--redo', dest='redo', action='store_true', 
                        help='redo an experiment')
    parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='gpu')
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)  # only default graph


def run_experiment(config, args, expt_id):
    # if train, switch the dataset to train, then
    #  train and save each model in the config spec
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)
    utils.write_config(config, os.path.join(config.working_dir, 'config.yaml'))

    print 'MAIN: parsing dataset'
    d = Dataset(config, config.base_dir)
    print 'MAIN: dataset done. took %.2fs' % (time.time() - start)

    if args.train:
        d.set_active_split(config.train_suffix)

        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue
            if args.model is not None and args.model != model_description['type']:
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
        results = []  # items to be written in executive summary 
        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue
            if args.model is not None and args.model != model_description['type']:
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
            results.append({
                'model-name': model_description['name'],
                'model-type': model_description['type'],
                'params': str(model_description['params']),
                'correlation': evaluation['mu_corr'],
                'performance': evaluation['mu_perf'],
                'model_dir': model_dir,
                'expt_id': expt_id
            })

            print 'MAIN: evaluation %s done, time %.2fs' % (
                model_description['name'], time.time() - start_time)

        return results

def validate_config(config):
    num_expts = config.num_experiments

    model_params = [
        x for m in config.model_spec \
        for x in (m['params'] or {}).values()
    ] + [
        v for k, v in config.vocab.iteritems()
    ]
    if not any(isinstance(x, list) for x in model_params):
        if num_expts != 1:
            print 'MAIN: falling back to 1 experiment...no randomizable values provided'
        return 1
    return num_expts


def validate_data(config):
    skipped_lines = 0
    data_spec = config.data_spec

    d = copy.deepcopy(dict(config._asdict()))
    in_data_prefix = os.path.join(config.data_dir, config.prefix)
    out_data_dir = os.path.join(config.working_dir, 'data')
    out_data_prefix = os.path.join(out_data_dir, config.prefix)
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    for split_suffix in [config.train_suffix, config.dev_suffix, config.test_suffix, '']:
        in_path = in_data_prefix + split_suffix
        assert os.path.exists(in_path), 'Split %s doesnt exist' % in_path

        out_path = out_data_prefix + '.validated' + split_suffix
        out_file = open(out_path, 'w')

        for l in open(in_path):
            parts = l.strip().split('\t')
            # invalid number of cells
            if len(data_spec) != len(parts): 
                skipped_lines += 1
                continue
            skip = False
            for x, var in zip(parts, data_spec):
                if var.get('skip', False): continue

                if var['type'] == 'continuous' and not utils.is_number(x):
                    skip = True; break
                if x == '':
                    skip = True; break
            if skip:
                skipped_lines += 1
                continue

            out_file.write(l)

        out_file.close()

    d['data_dir'] = out_data_dir
    d['prefix'] = d['prefix'] + '.validated'
    return namedtuple("config", d.keys())(**d), skipped_lines


def generate_experiment(parent_config, expt_id):
    """ choose a random thing from each config list 
    """
    d = copy.deepcopy(dict(parent_config._asdict()))

    assert not d['working_dir'].endswith('/')
    d['working_dir'] = d['working_dir'] + '_%s' % expt_id

    d['base_dir'] = parent_config.working_dir

    for k, v in d['vocab'].items():
        if isinstance(v, list):
            d['vocab'][k] = random.choice(v)

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
    if args.model is not None:
        num_experiments = 1
    reload(sys)
    sys.setdefaultencoding('utf8')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(config.seed)
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    print 'MAIN: validating data...'
    start = time.time()
    config, skipped = validate_data(config)
    print '\t done. Took %.2fs, found %d invalid rows' % (
        time.time() - start, skipped)

    # use config to preprocess data
    start = time.time()
    summary_path = os.path.join(config.working_dir, 'summary.csv')
    try:
        results = None
        for i in range(num_experiments):
            summary_file = open(summary_path, 'a')
            csv_writer = csv.writer(summary_file)

            expt = generate_experiment(config, i) if not args.redo else config
            if not args.redo and os.path.exists(os.path.join(expt.working_dir, 'config.yaml')):
                print 'MAIN: skipping expt ', i
                summary_file.close()
                continue
            results = run_experiment(expt, args, i)

            if i == 0:
                csv_writer.writerow(results[0].keys())
            print 'MAIN: writing summary to ', summary_path
            for res in results:
                csv_writer.writerow(res.values())
            summary_file.close()

    except:
        print 'MAIN: stopped with exception'
        traceback.print_exc()
    finally:
        pass

    # TODO maybe some kind of cleanup of temporrary files? like
    # datasets, etc etc
    utils.cleanup()

