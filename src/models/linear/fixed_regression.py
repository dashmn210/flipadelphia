
import regression_base
import plain_regression
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import pandas as pd
import numpy as np

class FixedRegression(plain_regression.RegularizedRegression):

    def _get_np_xy(self, dataset, target_name=None, level=None, features=None):
        X, y, features = plain_regression.RegularizedRegression._get_np_xy(
            self, dataset, target_name, level, features)

        # add 1-hot confounds to features
        for varname in self.confound_names:
            one_hots = dataset.np_data[dataset.split][varname]
            for var_level, col_idx in dataset.class_to_id_map[varname].items():
                new_col = np.reshape(one_hots[:,col_idx], (-1, 1))
                X = np.append(X, new_col, axis=1)
                features.append('%s|%s' % (varname, var_level))

        return X, y, features



