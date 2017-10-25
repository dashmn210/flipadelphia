
import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn
import numpy as np

class RegularizedRegression(regression_base.Regression):

    def __init__(self, config, params):
        regression_base.Regression.__init__(self, config, params)
        self.alpha = 1 if self.params['regularizor'] == 'l1' else 0
        self.lmbda = self.params['lambda']
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None


    def _data_to_numpy(self, dataset, target, ignored_vars, level=''):
        # note that cols are sorted just like self.features
        df = dataset.to_pd_df()

        if level is not '':
            # turn response into 1's on the favored level, and 0 elsewhere
            df = self._make_binary(df, target['name'], level)

        y = df[target['name']].as_matrix()

        # this also drops confounds
        not_in_covariates = \
            [target['name']] + \
            [v['name'] for v in ignored_vars]
        X = df.drop(not_in_covariates, axis=1)

        features = list(X.columns)
        assert features == dataset.features

        return y, X.as_matrix(), features


    def _fit_regression(self, dataset, target):
        X, y, features = self._get_np_xy(dataset, target['name'])

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


    def _fit_classifier(self, dataset, target, level=None):
        X, y, features = self._get_np_xy(dataset, target['name'], level)

        print 'REGRESSION: fitting target %s, level %s' % (target['name'], level)
        model_fitter = sklearn.linear_model.LogisticRegression(
            penalty=(self.regularizor or 'l2'),
            C=(1.0/self.lmbda) if self.regularizor > 0 else 99999999)
        model = model_fitter.fit(X, list(y))

        return regression_base.ModelResult(
            model=model,
            weights=self._sklearn_weights(model, features),
            response_type='categorical')


    def _get_np_xy(self, dataset, target_name, level=None):
        split = dataset.split
        X = dataset.np_data[split][dataset.input_varname()]
        y = dataset.np_data[split][target_name]
        if level is not None:
            target_col = dataset.class_to_id_map[target_name][level]
            y = y[:,target_col]
        y = np.squeeze(y) # stored as column even if just floats
        return X, y, dataset.ordered_features



    def _sklearn_weights(self, model, feature_names):
        weights = {}
        for w, f in zip(model.coef_, feature_names):
            weights[f] = w
        weights['intercept'] = model.intercept_
        return weights


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        for i, target in enumerate(self.targets):
            if target['type']== 'continuous':
                fitting_function = self._fit_regression
            else:
                fitting_function = partial(
                    self._fit_ovr, model_fitting_fn=self._fit_classifier)

            self.models[target['name']] = fitting_function(
                dataset=dataset,
                target=target)





