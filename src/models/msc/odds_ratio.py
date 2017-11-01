import sys
sys.path.append('../..')

from collections import defaultdict, namedtuple
import rpy2.robjects
from rpy2.robjects import r, pandas2ri
import src.msc.utils as utils
import src.msc.utils as utils
import math
import pickle
import os
import numpy as np
import time
from functools import partial
from tqdm import tqdm

from src.models.abstract_model import Model, Prediction
from selection_model import SelectionModel

class OddsRatio(SelectionModel):
    def __init__(self, config, params):
        SelectionModel.__init__(self, config, params)


    def _compute_ratios(self, dataset, target_name, feature_indices, level=None):
        """ computes odds ratios, returning a dict of each feature's ratio
            uses a subset of features, defined by feature_indices
        """
        if not level:
            # bucket based on bottom/top 30%
            response = np.copy(dataset.np_data[dataset.split][target_name].toarray())
            low_threshold = utils.percentile(response, 0.3)
            high_threshold = utils.percentile(response, 0.7)
            response[response < low_threshold] = 0
            response[response > high_threshold] = 1
        else:
            level_idx = dataset.class_to_id_map[target_name][level]
            response = dataset.np_data[dataset.split][target_name][:, level_idx].toarray()

        feature_indices = set(feature_indices)
        feature_counts = defaultdict(lambda: {0: 0, 1: 0})

        covariates = dataset.np_data[dataset.split][dataset.input_varname()]
        rows, cols = covariates.nonzero()
        for example, feature_idx in zip(rows, cols):
            if not feature_idx in feature_indices:  continue
            if not response[example][0] in [0, 1]: continue

            feature = dataset.ids_to_features[feature_idx]
            feature_counts[feature][response[example][0]] += 1

        ratios = {}
        for feature, counts in feature_counts.iteritems():
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/
            a = feature_counts[feature][0]
            b = feature_counts[feature][1]
            c = len(response[response == 0]) - a
            d = len(response[response == 1]) - b
            try:
                ratios[feature] = float(a * d) / (b * c)
            except ZeroDivisionError:
                pass

        return ratios


    def _select_features(self, dataset, model_dir):
        start = time.time()
        print 'ODDS_RATIO: selecting initial featureset'
        # {feature: [odds ratio for each confound]}
        feature_ratios = defaultdict(list)
        for var in tqdm(self.confounds):
            if var['type'] == 'categorical':
                for level in tqdm(dataset.class_to_id_map[var['name']]):
                    ratios = self._compute_ratios(
                        dataset=dataset,
                        target_name=var['name'],
                        level=level,
                        feature_indices=dataset.ids_to_features.keys())
            else:
                ratios = self._compute_ratios(
                    dataset=dataset,
                    target_name=var['name'],
                    feature_indices=dataset.ids_to_features.keys())
            for f, r in ratios.items():
                feature_ratios[f].append(r)

        feature_importance = sorted(
            map(lambda (f, ors): (np.mean(ors), f), feature_ratios.items()))
        # write this to output
        with open(os.path.join(model_dir, 'odds-ratio-scores-before-selection.txt'), 'w') as f:
            s = '\n'.join('%s\t%s' % (f, str(o)) for o, f in feature_importance)
            f.write(s)

        # choose K features with smallest odds ratio
        selected_features = feature_importance[:self.params['selection_features']]
        selected_features = map(lambda (ors, f): f, selected_features)
        print '\n\tdone. Took %.2fs' % (time.time() - start)
        return selected_features

