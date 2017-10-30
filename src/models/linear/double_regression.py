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
        self.residuals = None # to be filled up between passes

    def _iter_minibatches(self, dataset, target_name=None, features=None, 
                                level=None, batch_size=None):
        plain_iterator = plain_regression.RegularizedRegression._iter_minibatches(
            self, dataset, target_name, None, level, batch_size)


        # TODO -- FIGURE THIS OUT!! 
        # need to 
        #    only give confounds for first training + inference
        #    only give text for 2nd training

        i = 0
        while True:
            start = i
            end = (i+batch_size if batch_size else None)

            if self.residuals is not None:
                X, X_features = dataset.text_X_chunk(features, start, end)
                y = self.residuals[target_name][start:end]
                yield X, y, X_features

            else:
                if target_name is not None:
                    y = dataset.y_chunk(target_name, level, start, end)
                else:
                    y = None
                X, X_features = dataset.nontext_X_chunk(
                    self.confound_names, start, end)

                yield X, y, X_features


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
        print "DOUBLE REGRESSION: first pass using confounds..."
        f = self.train_model(dataset)
        self.models = f
        start = time.time()
        print "DOUBLE REGRESSION: inference for residuals..."
        preds = self.inference(dataset, model_dir).scores
        print "\tDone. Took %.2fs" % (time.time() - start)
        self.residuals = {}
        for i, target in enumerate(self.targets):
            y_hat = preds[target['name']]
            y = dataset.np_data[dataset.split][target['name']]
            self.residuals[target['name']] = y - y_hat

        print "DOUBLE REGRESSION: 2nd pass using text and residuals..."
        g = self.train_model(dataset, features=None)
        self.models = g

