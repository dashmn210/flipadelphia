import regression_base
from functools import partial

from rpy2.robjects import r, pandas2ri
import rpy2.robjects
import numpy as np
from collections import defaultdict
import pandas as pd

class MixedRegression(regression_base.Regression):
    def __init__(self, config, params):
        regression_base.Regression.__init__(self, config, params)
        self.confounds = []
        self.extra_ignored = []
        for var in self.config.data_spec[1:]:
            if var['control']:
                self.extra_ignored.append(var)
                if var['type'] == 'continuous':
                    print 'WARNING: mixed regression is skipping confound %s (continuous)' % (
                        var['name'])
                else:
                    self.confounds.append(var)

    def _get_np_xy(self, dataset, target_name, level=None):
        split = dataset.split
        y = dataset.np_data[split][target_name]
        if level is not None:
            target_col = dataset.class_to_id_map[target_name][level]
            y = y[:,target_col]
        y = np.squeeze(y) # stored as column even if just floats

        # now add confounds (NOT one-hot) to features
        X = dataset.np_data[split][dataset.input_varname()]
        features = dataset.ordered_features
        idx_to_class = np.vectorize(lambda idx: dataset.id_to_class_map[idx])
        for var in self.confounds:
            # undo one-hot stuff
            one_hots = dataset.np_data[split][var['name']]
            levels = np.argmax(one_hots, axis=1)
            new_col = idx_to_class(levels)
            new_col = np.reshape(new_col, (-1, 1))  # turn into a column
            X = np.append(X, new_col, axis=1)
            features.append(var['name'])

        return X, y, features


    def _fit_mixed_regression(self, dataset, target, ignored_vars):
        r_df_name = 'df_' + target['name']
        r_model_name = 'model_' + target['name']

        X, y, features = self._get_np_xy(dataset, target['name'])
        data = np.append(np.reshape(y, (-1,1)), X, 1)
        col_names = [target['name']] + features
        df = pd.DataFrame(
            data=data,
            columns=col_names)
        print df.dtypes
        # TODO keep everything but catigorical confounds as float
        # (categorical confounds should be )
        quit()
        # TDO -- make pd df from this 

        # TODO - handle this feature gracefully
        if '...' in set(df.columns):
            df.drop('...', axis=1, inplace=True)
        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        cmd = "lmer(%s, data=%s, REML=FALSE)"
        formula = '%s ~ . %s %s' % (
            target['name'],
            ''.join(' + (1|%s)' % confound['name'] for confound in self.confounds),
            ''.join(' - %s' % var['name'] for var in ignored_vars + self.confounds))
        print "MIXED: fitting ", cmd % (formula, r_df_name)
        model = r(cmd % (formula, r_df_name))
        rpy2.robjects.globalenv[r_model_name] = model

        params = self._extract_r_params(r_model_name)

        return regression_base.ModelResult(
            model=model,
            weights=params,
            response_type='continuous')


    def _fit_mixed_classifier(self, dataset, target, ignored_vars, level=''):
        r_df_name = 'df_%s_%s' % (target['name'], level)
        r_model_name = 'model_%s_%s' % (target['name'], level)

        df = dataset.to_pd_df()
        # otherwise the datset is assumed to be binary
        if level is not '':
            df = self._make_binary(df, target['name'], level)
        # TODO - handle this feature gracefully
        if '...' in set(df.columns):
            df.drop('...', axis=1, inplace=True)
        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        cmd = "glmer(%s, family=binomial(link='logit'), data=%s, REML=FALSE)"
        formula = '%s ~ . %s %s' % (
            target['name'],
            ''.join(' + (1|%s)' % confound['name'] for confound in self.confounds),
            ''.join(' - %s' % var['name'] for var in ignored_vars + self.confounds))

        print "MIXED: fitting ", cmd % (formula, r_df_name)
        model = r(cmd % (formula, r_df_name))
        rpy2.robjects.globalenv[r_model_name] = model

        params = self._extract_r_params(r_model_name)

        return regression_base.ModelResult(
            model=model,
            weights=params,
            response_type='categorical')


    def _extract_r_params(self, model_name):
        s = r("coef(%s)" % model_name)
        out = defaultdict(float)
        for i, x in enumerate(s[0].names):
            if 'intercept' in x.lower():
                x = 'intercept'
            out[x] = np.mean(s[0][i])
        return out


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        for i, target in enumerate(self.targets):
            ignored = self.targets[:i] + self.targets[i+1:]

            if target['type']== 'continuous':
                fitting_function = self._fit_mixed_regression
            else:
                fitting_function = partial(
                    self._fit_ovr, model_fitting_fn=self._fit_mixed_classifier)

            self.models[target['name']] = fitting_function(
                dataset=dataset,
                target=target,
                ignored_vars=ignored + self.extra_ignored)




