
import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn

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



    def _fit_regression(self, dataset, target, ignored_vars):
        r_df_name = 'df_' + target['name']
        r_model_name = 'model_' + target['name']

        y, X, feature_names = self._data_to_numpy(dataset, target, ignored_vars)

        if self.regularizor:
            if self.regularizor == 'l1':
                model_fitter = sklearn.linear_model.Lasso(alpha=self.lmbda)
            else:
                model_fitter = sklearn.linear_model.Ridge(alpha=self.lmbda)
        else:
            model_fitter = sklearn.linear_model.LinearRegression()

        print 'REGRESSION: fitting target %s'
        model = model_fitter.fit(X, y)

        weights = {}
        for w, f in zip(model.coef_, feature_names):
            weights[f] = w
        weights['intercept'] = model.intercept_

        return regression_base.ModelResult(
            model=model,
            weights=weights,
            response_type='continuous')


    def _fit_classifier(self, dataset, target, ignored_vars, level=''):
        r_df_name = 'df_%s_%s' % (target['name'], level)
        r_model_name = 'model_%s_%s' % (target['name'], level)

        y, X, feature_names = self._data_to_numpy(dataset, target, ignored_vars, level)

        model_fitter = sklearn.linear_model.LogisticRegression(
            penalty=(self.regularizor or 'l2'),
            C=(1.0/self.lmbda) if self.regularizor > 0 else 99999999)
        print 'REGRESSION: fitting target %s, level %s' % (target['name'], level)
        model = model_fitter.fit(X, list(y))
        weights = {}
        for w, f in zip(model.coef_[0], feature_names):
            weights[f] = w
        weights['intercept'] = model.intercept_

        return regression_base.ModelResult(
            model=model,
            weights=weights,
            response_type='categorical')



    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        for i, target in enumerate(self.targets):
            ignored = self.targets[:i] + self.targets[i+1:]
            # ignore the confounds as well as off-target targets
            ignored += self.confounds

            if target['type']== 'continuous':
                fitting_function = self._fit_regression
            else:
                fitting_function = partial(
                    self._fit_ovr, model_fitting_fn=self._fit_classifier)

            self.models[target['name']] = fitting_function(
                dataset=dataset,
                target=target,
                ignored_vars=ignored)





