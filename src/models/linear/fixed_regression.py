
import regression_base
import plain_regression
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import pandas as pd
import numpy as np
from scipy import sparse


def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a


class FixedRegression(plain_regression.RegularizedRegression):

    def _get_np_xy(self, dataset, target_name=None, level=None, features=None):
        X, y, features = plain_regression.RegularizedRegression._get_np_xy(
            self, dataset, target_name, level, features)

        # add 1-hot confounds to features
        new_cols = []
        for varname in self.confound_names:
            one_hots = dataset.np_data[dataset.split][varname]
            for var_level, col_idx in dataset.class_to_id_map[varname].items():
                new_col = np.reshape(one_hots[:,col_idx].toarray(), (-1, 1))
                new_cols.append(new_col)
                features.append('%s|%s' % (varname, var_level))

        X = sparse.hstack([X] + new_cols).tocsr()
        assert isinstance(X, sparse.csr.csr_matrix)
        return X, y, features



