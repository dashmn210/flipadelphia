
import regression_base
import plain_regression
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import pandas as pd
import numpy as np

class FixedRegression(plain_regression.RegularizedRegression):

    def _get_np_xy(self, dataset, target_name, level=None):
        split = dataset.split
        y = dataset.np_data[split][target_name]
        if level is not None:
            target_col = dataset.class_to_id_map[target_name][level]
            y = y[:,target_col]
        y = np.squeeze(y) # stored as column even if just floats

        # now add confounds to features
        X = dataset.np_data[split][dataset.input_varname()]
        features = dataset.ordered_features
        for var in self.confounds:
            one_hots = dataset.np_data[split][var['name']]
            for level, col_idx in dataset.class_to_id_map[var['name']].items():
                new_col = np.reshape(one_hots[:,col_idx], (-1, 1))
                X = np.append(X, new_col, axis=1)
                features.append('%s|%s' % (var['name'], level))

        return X, y, features


    def _data_to_numpy(self, dataset, target, ignored_vars, level=''):
        if level is not '':
            # turn response into 1's on the favored level, and 0 elsewhere
            df = self._make_binary(df, target['name'], level)

        df_confounds = df[[c['name'] for c in self.confounds]]

        y = df[target['name']].as_matrix()

        # this also drops confounds
        not_in_covariates = \
            [target['name']] + \
            [v['name'] for v in ignored_vars]
        X = df.drop(not_in_covariates, axis=1)

        # HACKY!!! :(
        # now add back in the confounds (1-hot if categorical)
        #  (and also modify self.features to know about these confounds)
        feature_names = list(X.columns)

        for confound in self.confounds:
            if confound['type'] == 'continuous':
                X = pd.concat([X, df_confounds[confound['name']]], axis=1)
            else:
                for level in dataset.class_to_id_map[confound['name']].keys():
                    new = self._make_binary(df_confounds, confound['name'], level)[confound['name']]
                    X = pd.concat([X, new], axis=1)   
                    # add level info to the name of the feature
                    X = X.rename(columns={
                        confound['name']: '%s_%s' % (confound['name'], level)
                        })

        features = list(X.columns)

        return y, X.as_matrix(), features

