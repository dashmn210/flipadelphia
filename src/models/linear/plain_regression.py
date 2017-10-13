
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


    def _fit_regression(self, split, dataset, target, ignored_vars, confounds):
        r_df_name = 'df_' + target['name']
        r_model_name = 'model_' + target['name']

        df = dataset.to_pd_df(split)
        y = df[target['name']].as_matrix()
        X = df.drop([target['name']] + [v['name'] for v in ignored_vars], axis=1)

        if self.regularizor:
            if self.regularizor == 'l1':
                model_fitter = sklearn.linear_model.Lasso(alpha=self.lmbda)
            else:
                model_fitter = sklearn.linear_model.Ridge(alpha=self.lmbda)
        else:
            model_fitter = sklearn.linear_model.LinearRegression()

        model = model_fitter.fit(X, y)

        weights = {}
        for w, f in zip(model.coef_, dataset.features):
            weights[f] = w

        return regression_base.Regression.Model(
            model=model,
            weights=weights,
            is_r=False)


    def _fit_classifier(self, split, dataset, target, ignored_vars, confounds, level=''):
        r_df_name = 'df_%s_%s' % (target['name'], level)
        r_model_name = 'model_%s_%s' % (target['name'], level)

        df = dataset.to_pd_df(split)

        # otherwise the datset is assumed to be binary
        if level is not '':
            df = self._make_binary(df, target['name'], level)

        y = df[target['name']].as_matrix()
        X = df.drop([target['name']] + [v['name'] for v in ignored_vars], axis=1)

        model_fitter = sklearn.linear_model.LogisticRegression(
            penalty=(self.regularizor or 'l2'),
            C=(1.0/self.lmbda) if self.regularizor > 0 else 99999999)

        model = model_fitter.fit(X, list(y))
        weights = {}
        for w, f in zip(model.coef_[0], dataset.features):
            weights[f] = w

        return regression_base.Regression.Model(
            model=model,
            weights=weights,
            is_r=False)



    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        train_split = self.config.train_suffix

        targets = [
            var for var in self.config.data_spec[1:] \
            if var['control'] == False \
            and not var.get('skip', False)]

        confounds = []

        for i, target in enumerate(targets):
            ignored_variables = [
                var for var in self.config.data_spec[1:] \
                if var['name'] != target['name'] \
                and not var.get('skip', False)]

            if target['type']== 'continuous':
                fitting_function = self._fit_regression
            else:
                fitting_function = partial(
                    self._fit_ovr, model_fitting_fn=self._fit_classifier)

            self.models[target['name']] = fitting_function(
                split=train_split,
                dataset=dataset,
                target=target,
                ignored_vars=ignored_variables,
                confounds=confounds)

        print self.models




