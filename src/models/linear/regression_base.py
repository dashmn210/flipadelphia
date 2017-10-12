"""
TODO -- refactor out one-vs-rest + reression stuff into seperate classes
         and use this as an actual wrapper


     -- BUCKET CONFOUNDS THAT ARE CONTINUOUS
"""


import sys
sys.path.append('../..')

from collections import defaultdict, namedtuple
import rpy2.robjects
from rpy2.robjects import r, pandas2ri

#r("options(warn=-1)").  # TODO -- figure out how to silence warnings like rank-deficient
r("library('lme4')") 
r("library(MuMIn)")
pandas2ri.activate()


rModel = namedtuple('rModel', 
    ('model', 'r_model_name', 'r_df_name'))


class Regression:

    def __init__(self, config, params):
        self.config = config
        self.params = params
        # target variable name: R object with this model  OR  list of one-vs-rest models, one per level
        self.models = {}


    def save(self, dir):
        """ saves a representation of the model into a directory
        """
        raise NotImplementedError


    def load(self, dataset, model_dir):
        """ creates or loads a model
        """
        pass


    def _fit(self, cmd, df, target_var, ignored_vars, confound_vars, name=''):
        """ cmd is a valid R fitting command, with unfilled %s's for
                the formula and data parts
            df is a pandas df with *all* variables (target + ignored + confounds) as cols
            target_var is the thing we want to predict
            ignored_vars is a list of variables we want to ignore
            confound_vars is a list of variables we want to control
            name is an optional suffix for the r environment
        """
        r_df_name = 'df_' + target_var['name'] + ('_%s' % name if name else '')
        r_model_name = 'model_' + target_var['name'] + ('_%s' % name if name else '')

        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)
        formula = '%s ~ %s %s' % (
            target_var['name'],
            ''.join(' + (1|%s)' % confound['name'] for confound in confound_vars),
            ''.join(' - %s' % var['name'] for var in ignored_vars + confound_vars))
        res = r(cmd % (formula, r_df_name))
        print '[regression_base]: fitting ', cmd % (formula, r_df_name)
        rpy2.robjects.globalenv[r_model_name] = res
        return rModel(
            model=res,
            r_model_name=r_model_name,
            r_df_name=r_df_name)



    def _fit_ovr(self, split, dataset, target, ignored_vars, confounds, model_fitting_fn):
        models = {}
        for level in dataset.class_to_id_map[target['name']].keys():
            models[level] = model_fitting_fn(
                split, dataset, target, ignored_vars, confounds, level=level)
        return models


    def _make_binary(self, df, col_name, selected_level):
        """ returns a copy of df where a categorical column (col_name)
             is set to 1 where examples are the selected_level and 0 otherwise
            TODO -- think of a better name for this
        """
        assert selected_level != 0, '%s shouldnt be 0' % selected_level

        out = df.copy(deep=True)
        # set off-selected to 0
        out.loc[df[col_name] != selected_level, col_name] = 0
        # set selected to 1
        out.loc[df[col_name] == selected_level, col_name] = 1

        return out


    def inference(self, dataset, model_dir, dev=True):
        """ run inference on the dev/test set, save all predictions to 
                per-variable files in model_dir, and return pointers to those files
            saves model-specific metrics/artifacts (loss, attentional scores, etc) 
                into self.report (also possible writes to a file in model_dir)
        """
        raise NotImplementedError


    def report(self):
        """ releases self.report, a summary of the last job this model
                executed whether that be training, testing, etc
        """
        raise NotImplementedError




