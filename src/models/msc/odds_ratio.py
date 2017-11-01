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


class OddsRatio(Model):
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
        self.confounds = [
            variable for variable in variables \
            if variable['control'] and not variable['skip']]


    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        models_file = os.path.join(model_dir, 'models')
        utils.pickle(self.models, models_file)
        print 'ODDS_RATIO: models saved into ', models_file


    def load(self, dataset, model_dir):
        start = time.time()
        self.models = utils.depickle(os.path.join(model_dir, 'models'))
        target_names = map(lambda x: x['name'], self.targets)
        assert set(target_names) == set(self.models.keys())
        print 'ODDS_RATIO: loaded model parameters from %s, time %.2fs' % (
            model_dir, time.time() - start)

    def _compute_ratios(self, dataset, target_name, ratios_dict, feature_indices, level=None):
        """ computes odds ratios, returning a dict of each feature's ratio
            uses a subset of features, defined by feature_indices
        """
        if not level:
            # bucket based on bottom/top 30%
            # TODO
            response = dataset.np_data[dataset.split][target_name]
            print response
            quit()
        else:
            level_idx = dataset.class_to_id_map[target_name][level]
            response = dataset.np_data[dataset.split][target_name][:, level_idx]
            print response; quit()

    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        print 'ODDS_RATIO: computing initial featureset'
        # {feature: [odds ratio for each confound]}
        feature_ratios = defaultdict(list)
        for var in self.confounds:
            if var['type'] == 'categorical':
                for level in dataset.class_to_id_map[var['name']]:
                    ratios = self._compute_ratios(
                        dataset=dataset,
                        target_name=var['name'],
                        level=level,
                        ratios_dict= feature_ratios,
                        feature_indices=dataset.ids_to_features.keys())
            else:
                ratios = self._compute_ratios(
                    dataset=dataset,
                    target_name=var['name'],
                    ratios_dict= feature_ratios,
                    feature_indices=dataset.ids_to_features.keys())
            for f, r in ratios.items():
                feature_ratios[f].append(r)


        for i, target in enumerate(self.targets):
            if target['type'] == 'continuous':
                self.models[target['name']] = self._fit_regression(
                    dataset=dataset, 
                    target=target,
                    features=features)
            else:
                self.models[target['name']] = self._fit_ovr(
                    dataset=dataset, 
                    target=target,
                    features=features)

