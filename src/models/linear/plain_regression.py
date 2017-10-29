
import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import numpy as np
from tqdm import tqdm
from scipy import sparse


class RegularizedRegression(regression_base.Regression):

    def __init__(self, config, params, intercept=True):
        regression_base.Regression.__init__(self, config, params, intercept)
        self.lmbda = self.params.get('lambda', 0)
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None
        self.batch_size = self.params['batch_size']


    def _iter_minibatches(self, X, y):
        assert isinstance(X, sparse.csr.csr_matrix)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

        i = 0
        while True:
            yield X[i:i+self.batch_size,:], y[i:i+self.batch_size]
            i += self.batch_size
            if i > len(X):
                i = 0


    def _sklearn_weights(self, model, feature_names):
        weights = {}
        for w, f in zip(np.squeeze(model.coef_), feature_names):
            weights[f] = w
        if self.use_intercept:
            weights['intercept'] = model.intercept_
        return weights


    def _fit_regression(self, dataset, target, features=None):
        X, y, features = self._get_np_xy(
            dataset=dataset,
            target_name=target['name'],
            features=features)

        print 'REGRESSION: fitting target %s' % target['name']
        model = sklearn.linear_model.SGDRegressor(
            penalty=self.regularizor or 'none',
            alpha=self.lmbda,
            learning_rate='constant',
            eta0=1.0)

        for _ in tqdm(range(self.params['num_train_steps'])):
            Xi, yi = next(self._iter_minibatches(X, y))
            model.partial_fit(Xi, yi)

        return regression_base.ModelResult(
            model=model,
            weights=self._sklearn_weights(model, features),
            response_type='continuous')


    def _fit_classifier(self, dataset, target, level=None, features=None):
        X, y, features = self._get_np_xy(
            dataset=dataset, 
            target_name=target['name'], 
            level=level, 
            features=features)

        print 'CLASSIFICATION: fitting target %s, level %s' % (target['name'], level)
        model = sklearn.linear_model.SGDClassifier(
            loss='log',
            penalty=(self.regularizor or 'none'),
            alpha=self.lmbda,
            learning_rate='constant',
            eta0=1.0)

        for _ in tqdm(range(self.params['num_train_steps'])):
            Xi, yi = next(self._iter_minibatches(X, y))
            model.partial_fit(Xi, yi, classes=np.unique(y))

        return regression_base.ModelResult(
            model=model,
            weights=self._sklearn_weights(model, features),
            response_type='categorical')

