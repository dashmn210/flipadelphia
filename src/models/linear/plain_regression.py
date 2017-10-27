
import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import numpy as np



class RegularizedRegression(regression_base.Regression):

    def __init__(self, config, params):
        regression_base.Regression.__init__(self, config, params)
        self.lmbda = self.params.get('lambda', 0)
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None


    def _fit_regression(self, dataset, target, features=None):
        X, y, features = self._get_np_xy(
            dataset=dataset,
            target_name=target['name'],
            features=features)

        print 'REGRESSION: fitting target %s' % target['name']
        if self.regularizor:
            if self.regularizor == 'l1':
                model_fitter = sklearn.linear_model.Lasso(alpha=self.lmbda)
            else:
                model_fitter = sklearn.linear_model.Ridge(alpha=self.lmbda)
        else:
            model_fitter = sklearn.linear_model.LinearRegression()
        model = model_fitter.fit(X, y)

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

        print 'REGRESSION: fitting target %s, level %s' % (target['name'], level)
        model_fitter = sklearn.linear_model.LogisticRegression(
            penalty=(self.regularizor or 'l2'),
            C=(1.0/self.lmbda) if self.regularizor > 0 else 99999999)
        model = model_fitter.fit(X, list(y))

        return regression_base.ModelResult(
            model=model,
            weights=self._sklearn_weights(model, features),
            response_type='categorical')


    def _sklearn_weights(self, model, feature_names):
        weights = {}
        for w, f in zip(np.squeeze(model.coef_), feature_names):
            weights[f] = w
        weights['intercept'] = model.intercept_
        return weights



