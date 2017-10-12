
import regression_base
from functools import partial

class MixedRegression(regression_base.Regression):


    def _fit_mixed_regression(self, split, dataset, target, ignored_vars, confounds):
        df = dataset.to_pd_df(split)
        cmd = "lmer(%s, data=%s, REML=FALSE)" 
        return self._fit(cmd, df, target, ignored_vars, confounds)


    def _fit_mixed_classifier(self, split, dataset, target, ignored_vars, confounds, level=''):
        df = dataset.to_pd_df(split)
        # otherwise the datset is assumed to be binary
        if level is not '':
            df = self._make_binary(df, target['name'], level)
        cmd = "glmer(%s, family=binomial(link='logit'), data=%s, REML=FALSE)"
        return self._fit(cmd, df, target, ignored_vars, confounds, name=level)


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        train_split = self.config.train_suffix

        targets = [
            variable for variable in self.config.data_spec[1:] \
            if variable['control'] == False \
            and not variable.get('skip', False)]
        confounds = [
            variable for variable in self.config.data_spec[1:] \
            if variable['control'] \
            and not variable.get('skip', False)]

        for i, target in enumerate(targets):
            ignored_targets = targets[:i] + targets[i+1:]

            if target['type']== 'continuous':
                fitting_function = self._fit_mixed_regression
            else:
                fitting_function = partial(
                    self._fit_ovr, model_fitting_fn=self._fit_mixed_classifier)

            self.models[target['name']] = fitting_function(
                split=train_split,
                dataset=dataset,
                target=target,
                ignored_vars=ignored_targets,
                confounds=confounds)

        print self.models




