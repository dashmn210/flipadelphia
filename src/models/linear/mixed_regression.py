import regression_base
from functools import partial
from eventlet import Timeout
from rpy2.robjects import r, pandas2ri
import rpy2.robjects
import numpy as np
from collections import defaultdict
import pandas as pd
import traceback

class MixedRegression(regression_base.Regression):
    def __init__(self, config, params, intercept=False):
        regression_base.Regression.__init__(self, config, params, intercept)
        self.confound_names = []
        for var in self.config.data_spec[1:]:
            if var['control'] and not var['skip']:
                if var['type'] == 'continuous':
                    print 'WARNING: mixed regression is skipping confound %s (continuous)' % (
                        var['name'])
                else:
                    self.confound_names.append(var['name'])

    def _extract_r_coefs(self, model_name):
        s = r("coef(%s)" % model_name)
        out = defaultdict(float)
        for i, x in enumerate(s[0].names):
            if 'intercept' in x.lower():
                x = 'intercept'
            if x == 'intercept' and not self.use_intercept:
                continue
            out[x] = np.mean(s[0][i])
        return out


    def _train_r_model(self, cmd, model_name):
        print "MIXED: fitting ", cmd
        timeout = Timeout(2)
        try:
            model = r(cmd)
            rpy2.robjects.globalenv[model_name] = model
            params = self._extract_r_coefs(model_name)
        except:
            print "MIXED: timed out or broke. Fuck you too R! filling model with 0's"
            print "MIXED: traceback:"
            print traceback.format_exc()
            model = None
            params = defaultdict(float)
        finally:
            timeout.cancel()

        return model, params


    def _get_pd_df(self, dataset, target_name, level=None):
        df = pd.DataFrame()

        # start with your targets
        split = dataset.split
        y = dataset.np_data[split][target_name]
        if level is not None:
            target_col = dataset.class_to_id_map[target_name][level]
            y = y[:,target_col].astype('str')
            target_name += '.' + level
        y = np.squeeze(y) # stored as column even if just floats
        df[target_name] = y

        # now add in text features
        X = dataset.np_data[split][dataset.input_varname()]
        features = dataset.ordered_features
        for occurances, feature in zip(X.T, features):
            df[feature] = occurances

        # now add in all the confounds (undoing 1-hot)
        for varname in self.confound_names:
            idx_to_class = np.vectorize(
                lambda idx: dataset.id_to_class_map[varname][idx])
            # undo one-hot stuff
            one_hots = dataset.np_data[split][varname]
            levels = np.argmax(one_hots, axis=1)
            new_col = idx_to_class(levels)
            new_col = np.reshape(new_col, (-1, 1))  # turn into a column
            df[varname] = new_col

        return df


    def _fit_regression(self, dataset, target, **kwargs):
        r_df_name = 'df_' + target['name']
        r_model_name = 'model_' + target['name']

        df = self._get_pd_df(dataset, target['name'])
        # TODO - handle this feature gracefully
        if '...' in set(df.columns):
            df.drop('...', axis=1, inplace=True)

        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        cmd = "lmer(%s, data=%s, REML=FALSE)"
        formula = '%s ~ . %s %s' % (
            target['name'],
            ''.join(' + (1|%s)' % varname for varname in self.confound_names),
            ''.join(' - %s' % varname for varname in self.confound_names))

        print "MIXED: fitting ", cmd % (formula, r_df_name)
        model, params = self._train_r_model(cmd % (formula, r_df_name), r_model_name)

        return regression_base.ModelResult(
            model=model,
            weights=params,
            response_type='continuous')


    def _fit_classifier(self, dataset, target, level=None, **kwargs):
        r_df_name = 'df_%s_%s' % (target['name'], level)
        r_model_name = 'model_%s_%s' % (target['name'], level)

        df = self._get_pd_df(dataset, target['name'], level)
        # TODO - handle this feature gracefully
        if '...' in set(df.columns):
            df.drop('...', axis=1, inplace=True)

        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        cmd = "glmer(%s, family=binomial(link='logit'), data=%s, REML=FALSE)"
        formula = '%s ~ . %s %s' % (
            '%s.%s' % (target['name'], level) if level else target['name'],
            ''.join(' + (1|%s)' % varname for varname in self.confound_names),
            ''.join(' - %s' % varname for varname in self.confound_names))

        model, params = self._train_r_model(cmd % (formula, r_df_name), r_model_name)

        return regression_base.ModelResult(
            model=model,
            weights=params,
            response_type='categorical')








