
import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects


class RegularizedRegression(regression_base.Regression):

    def __init__(self, config, params):
        regression_base.Regression.__init__(self, config, params)
        self.alpha = 1 if self.params['regularizor'] == 'l1' else 0
        self.lmbda = self.params['lambda']


    def _fit_regression(self, split, dataset, target, ignored_vars, confounds):
        r_df_name = 'df_' + target['name']
        r_model_name = 'model_' + target['name']

        df = dataset.to_pd_df(split)
        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        covariate_matrix = "data.matrix(%s[, names(%s) != '%s'])" % (
            r_df_name, r_df_name, target['name'])
        response_vector = "%s[,'%s']" % (
            r_df_name, target['name'])

        cmd = "glmnet(%s, %s, alpha=%d, lambda=%d, family='gaussian')" % (
            covariate_matrix, response_vector, self.alpha, self.lmbda)
        model = r(cmd)
        rpy2.robjects.globalenv[r_model_name] = model

        return regression_base.rModel(
            model=model,
            r_model_name=r_model_name,
            r_df_name=r_df_name)



    def _fit_classifier(self, split, dataset, target, ignored_vars, confounds, level=''):
        r_df_name = 'df_%s_%s' % (target['name'], level)
        r_model_name = 'model_%s_%s' % (target['name'], level)

        df = dataset.to_pd_df(split)

        # otherwise the datset is assumed to be binary
        if level is not '':
            df = self._make_binary(df, target['name'], level)

        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        covariate_matrix = "data.matrix(%s[, names(%s) != '%s'])" % (
            r_df_name, r_df_name, target['name'])
        response_vector = "%s[,'%s']" % (
            r_df_name, target['name'])

        cmd = "glmnet(%s, %s, alpha=%d, lambda=%d, family='binomial')" % (
            covariate_matrix, response_vector, self.alpha, self.lmbda)
        model = r(cmd)
        rpy2.robjects.globalenv[r_model_name] = model

        return regression_base.rModel(
            model=model,
            r_model_name=r_model_name,
            r_df_name=r_df_name)


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




