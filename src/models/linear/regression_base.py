"""
TODO -- refactor out one-vs-rest + reression stuff into seperate classes
         and use this as an actual wrapper


     -- BUCKET CONFOUNDS THAT ARE CONTINUOUS


     -- FUCK R!!! so just pull out params after training
            and use them raw from then on
"""


import sys
sys.path.append('../..')

from collections import defaultdict, namedtuple
import rpy2.robjects
from rpy2.robjects import r, pandas2ri
from src.models.abstract_model import Model, Prediction
import src.msc.utils as utils
import src.msc.utils as utils
import math
import pickle
import os
import numpy as np
import time
from functools import partial

#r("options(warn=-1)").  # TODO -- figure out how to silence warnings like rank-deficient
r("library('lme4')") 
r("library('MuMIn')")
r("library('glmnet')")
pandas2ri.activate()


ModelResult = namedtuple('ModelResult', 
    ('model', 'response_type', 'weights'))


class Regression(Model):
    """ base class for all regression-type models

    """
    def __init__(self, config, params):
        Model.__init__(self, config, params)
        # target variable name (exploded if categorical)
        #     maps to ===>  R object with this model  
        self.models = {}

        variables = [v for v in self.config.data_spec[1:] \
                        if not v.get('skip', False)]
        self.targets = [
            variable for variable in variables \
            if variable['control'] == False and not variable['skip']]
        self.confound_names = [
            variable['name'] for variable in variables \
            if variable['control'] and not variable['skip']]


    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        models_file = os.path.join(model_dir, 'models')
        utils.pickle(self.models, models_file)
        print 'REGRESSION: models saved into ', models_file


    def load(self, dataset, model_dir):
        start = time.time()
        self.models = utils.depickle(os.path.join(model_dir, 'models'))
        target_names = map(lambda x: x['name'], self.targets)
        assert set(target_names) == set(self.models.keys())
        print 'REGRESSION: loaded model parameters from %s, time %.2fs' % (
            model_dir, time.time() - start)


    def _summarize_model_weights(self):
        def nested_model_iter(d):
            for _, model in d.iteritems():
                if isinstance(model, dict):
                    for k, v in nested_model_iter(model):
                        yield k, v
                else:
                    for k, v in model.weights.items():
                        yield k, v

        weights = defaultdict(list)
        # get duplicate of self.models except lists of weights
        for feature, value in nested_model_iter(self.models):
            weights[feature].append(value)
        out = {
            f: np.mean(v) for f, v in weights.iteritems()
        }
        return out


    def inference(self, dataset, model_dir):
        X, _, features = self._get_np_xy(dataset)

        predictions = defaultdict(dict)
        for response_name, val in self.models.iteritems():
            if isinstance(val, dict):
                # convert {level: scores} to 2d matrix with rows like:
                #  level1 score, level2 score, etc
                # (where ordering is determined by the dataset)
                response_levels = dataset.num_levels(response_name)
                level_predictions = \
                    lambda level: self._predict(X, features, val[dataset.id_to_class_map[response_name][level]])
                arr = np.array(
                    [level_predictions(l) for l in range(response_levels)])
                predictions[response_name] = np.transpose(arr, [1, 0])
            else:
                predictions[response_name] = self._predict(X, features, val)

        average_coefs = self._summarize_model_weights()

        return Prediction(
            scores=predictions,
            feature_importance=average_coefs)


    def _predict(self, X, feature_names, model):
        def score(example):
            s = 0
            for xi, feature in zip(example, feature_names):
                s += model.weights.get(feature, 0) * xi
            return s + model.weights['intercept']

        out = []
        for row in X:
            s = score(row)
            if model.response_type == 'continuous':
                out.append(s)
            else:
                out.append(1.0 / math.exp(-s))
        return out


    def _get_np_xy(self, dataset, target_name=None, level=None):
        split = dataset.split
        X = dataset.np_data[split][dataset.input_varname()]

        if not target_name:
            return X, None, dataset.ordered_features

        y = dataset.np_data[split][target_name]
        if level is not None:
            target_col = dataset.class_to_id_map[target_name][level]
            y = y[:,target_col]
        y = np.squeeze(y) # stored as column even if just floats
        return X, y, dataset.ordered_features


    def _fit_regression(self, dataset, target, ignored_vars):
        raise NotImplementedError


    def _fit_classifier(self, dataset, target, ignored_vars, level=''):
        raise NotImplementedError



    def _fit_ovr(self, dataset, target, model_fitting_fn):
        models = {}
        for level in dataset.class_to_id_map[target['name']].keys():
            models[level] = model_fitting_fn(
                dataset, target, level=level)
        return models


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        for i, target in enumerate(self.targets):
            if target['type'] == 'continuous':
                self.models[target['name']] = self._fit_regression(
                    dataset=dataset, 
                    target=target)
            else:
                self.models[target['name']] = self._fit_ovr(
                    dataset=dataset, 
                    target=target, 
                    model_fitting_fn=self._fit_classifier)



