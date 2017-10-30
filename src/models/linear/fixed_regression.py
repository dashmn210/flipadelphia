
import regression_base
import plain_regression
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import pandas as pd
import numpy as np
from scipy import sparse



class FixedRegression(plain_regression.RegularizedRegression):

    def _iter_minibatches(self, dataset, target_name=None, features=None, 
                                level=None, batch_size=None):
        i = 0
        while True:
            yield dataset.chunk(
                target_name=target_name,
                target_level=level,
                text_feature_subset=features,
                start=i,
                end=(i+batch_size if batch_size else None),
                aux_features=self.confound_names)  # new!!!

            if batch_size is None:
                break

            i += batch_size
            if i + batch_size > dataset.split_sizes[dataset.split]:
                i = 0



