
import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects

import sklearn

class FixedRegression(plain_regression.RegularizedRegression):

    def __init__(self, config, params):
        regression_base.Regression.__init__(self, config, params)
        self.alpha = 1 if self.params['regularizor'] == 'l1' else 0
        self.lmbda = self.params['lambda']
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None
        # TODO -- PARSE TARGETS AND STUFF UP HERE!!!


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




