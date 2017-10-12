
import regression_base
from functools import partial

class RegularizedRegression(regression_base.Regression):

    def __init__(self, config, params):
        regression_base.Regression.__init__(self, config, params)
        self.alpha = 1 if self.params['regularizor'] == 'l1' else 0
        self.lmbda = self.params['lambda']


    def _fit_regression(self, split, dataset, target, ignored_vars, confounds):
        df = dataset.to_pd_df(split)
        # TODO -- FIND A PACKAGE THAT DOES RIDGE/LASSO
        #         WITH FORMULAS!!!

        cmd = "glmnet(%s, data=%s, alpha=" + str(self.alpha) + ", lambda=" + str(self.lmbda) + ")"
        return self._fit(cmd, df, target, ignored_vars, confounds)


    def _fit_classifier(self, split, dataset, target, ignored_vars, confounds, level=''):
        df = dataset.to_pd_df(split)
        # otherwise the datset is assumed to be binary
        if level is not '':
            df = self._make_binary(df, target['name'], level)
        print 'HERE'
#        cmd = "glmer(%s, family=binomial(link='logit'), data=%s, REML=FALSE)"
        return self._fit(cmd, df, target, ignored_vars, confounds, name=level)


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
            print ignored_variables

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




