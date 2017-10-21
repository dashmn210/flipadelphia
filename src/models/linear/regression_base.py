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
            if variable['control'] == False]
        self.confounds = [
            variable for variable in variables \
            if variable['control']]


    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for response_name, rmodel in self.models.iteritems():
            response_file = os.path.join(model_dir, response_name)
            utils.pickle(rmodel, response_file)
        print 'REGRESSION: saved into ', model_dir


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
        df = dataset.to_pd_df()
        predictions = defaultdict(dict)
        for response_name, val in self.models.iteritems():
            if isinstance(val, dict):
                # convert {level: scores} to 2d matrix with rows like:
                #  level1 score, level2 score, etc
                # (where ordering is determined by the dataset)
                response_levels = dataset.num_levels(response_name)
                level_predictions = \
                    lambda level: self._predict(df, val[dataset.id_to_class_map[level]])
                arr = np.array(
                    [level_predictions(l) for l in range(response_levels)])
                predictions[response_name] = np.transpose(arr, [1, 0])
            else:
                predictions[response_name] = self._predict(df, val)

        average_coefs = self._summarize_model_weights()

        return Prediction(
            scores=predictions,
            feature_importance=average_coefs)


    def _predict(self, df, model):
        def score(example):
            s = 0
            for f, w in model.weights.items():
                s += float(example.get(f, 0) * w or 0)
            return s + model.weights['intercept']

        out = []
        for _, row in df.iterrows():
            s = score(row)
            if model.response_type == 'continuous':
                out.append(s)
            else:
                out.append(1.0 / math.exp(-s))
        return out


    def load(self, dataset, model_dir):
        start_time = time.time()
        target_names = map(lambda x: x['name'], self.targets)
        for filename in os.listdir(model_dir):
            if filename not in target_names:
                continue
            self.models[filename] = \
                utils.depickle(os.path.join(model_dir, filename))

        assert set(target_names) == set(self.models.keys())
        print 'REGRESSION: loaded model parameters from %s, time %.2fs' % (
            model_dir, time.time() - start)


    def _fit_regression(self, dataset, target, ignored_vars):
        raise NotImplementedError

    def _fit_classifier(self, dataset, target, ignored_vars, level=''):
        raise NotImplementedError


    def train(self, dataset, model_dir):
        raise NotImplementedError


    def _fit_ovr(self, dataset, target, ignored_vars, model_fitting_fn):
        models = {}
        for level in dataset.class_to_id_map[target['name']].keys():
            models[level] = model_fitting_fn(
                dataset, target, ignored_vars, level=level)
        return models


    def _make_binary(self, df, col_name, selected_level):
        """ returns a copy of df where a categorical column (col_name)
             is set to 1 where examples are the selected_level and 0 otherwise
            TODO -- think of a better name for this
        """
        assert selected_level != 0, '%s shouldnt be 0' % selected_level

        out = df.copy(deep=True)
        # set off-selected to 0
        out.loc[df[col_name] != selected_level, col_name] = 0
        # set selected to 1
        out.loc[df[col_name] == selected_level, col_name] = 1

        return out

    
