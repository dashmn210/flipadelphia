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
import math
import pickle
import os
import numpy as np

#r("options(warn=-1)").  # TODO -- figure out how to silence warnings like rank-deficient
r("library('lme4')") 
r("library('MuMIn')")
r("library('glmnet')")
pandas2ri.activate()


ModelResult = namedtuple('ModelResult', 
    ('model', 'response_type', 'weights'))


class Regression(Model):

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

    def _summarize_model_weights(self):
        def dict_average(dict_list):
            if len(dict_list) > 1:
                assert all(
                    set(dict_list[0]) == set(dict_list[i])\
                    for i in range(len(dict_list))[1:])
            return {
                f: sum(d[f] for d in dict_list) / len(dict_list) \
                for f in dict_list[0].keys()
            }

        weights = {}
        # get duplicate of self.models except lists of weights
        for k, v in self.models.items():
            if isinstance(v, dict):
                if k not in weights:
                    weights[k] = {}
                for k2, v2 in v.items():
                    if k2 not in weights[k]:
                        weights[k][k2] = []
                    weights[k][k2].append(v2.weights)
            else:
                if k not in weights:
                    weights[k] = []
                weights[k].append(v.weights)

        # now average each list
        out = defaultdict(dict)
        for k, v in weights.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    out[k][k2] = dict_average(v2)
            else:
                out[k] = dict_average(v)
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
        target_names = map(lambda x: x['name'], self.targets)
        for filename in os.listdir(model_dir):
            if filename not in target_names:
                continue
            self.models[filename] = \
                utils.depickle(os.path.join(model_dir, filename))

        assert set(target_names) == set(self.models.keys())
        print 'INFO: loaded model parameters from ', model_dir


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

    
