"""
f(C, D) => y
g(T) => residual

"""

import sys
sys.path.append('../..')
import regression_base
import plain_regression

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


class DoubleRegression(plain_regression.RegularizedRegression):

    def __init__(self, config, params):
        plain_regression.RegularizedRegression.__init__(self, config, params)
        self.lmbda = self.params.get('lambda', 0)
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None

    # replace with iterator?
    def _get_np_xy(self, dataset, target_name, level=None, features=None):
        # TODO -- from here!!!!
        X, y, f = plain_regression.RegularizedRegression._get_np_xy(
            self, dataset, target_name, level, features=None)
        if not features:
            return X, y, f

        # we're in the first round, where we only use confounds
        X = None
        features = []
        for varname in features:
            for var_level, col_idx in dataset.class_to_id_map[varname].items():
                new_col = np.reshape(one_hots[:,col_idx], (-1, 1))
                X = new_col if not X else np.append(X, new_col, axis=1)
                features.append('%s|%s' % (varname, var_level))

        print X
        print y
        print features; quit()
        return X, y, features                

    def train_model(self, dataset, features=None):
        models = {}
        for i, target in enumerate(self.targets):
            if target['type'] == 'continuous':
                models[target['name']] = plain_regression.RegularizedRegression._fit_regression(
                    self,
                    dataset=dataset, 
                    target=target,
                    features=features)
            else:
                models[target['name']] = plain_regression.RegularizedRegression._fit_ovr(
                    self,
                    dataset=dataset, 
                    target=target,
                    features=features)
        return models

    def train(self, dataset, model_dir):
        print self.confound_names
        f = self.train_model(dataset, features=self.confound_names)


