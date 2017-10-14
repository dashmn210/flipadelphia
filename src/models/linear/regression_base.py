"""
TODO -- refactor out one-vs-rest + reression stuff into seperate classes
         and use this as an actual wrapper


     -- BUCKET CONFOUNDS THAT ARE CONTINUOUS


     -- FUCK R!!! so just pull out params after training
            and use them raw from then on
"""


import sys
sys.path.append('../..')

from collections import defaultdict, namedtuple
import rpy2.robjects
from rpy2.robjects import r, pandas2ri
from src.models.abstract_model import Model

#r("options(warn=-1)").  # TODO -- figure out how to silence warnings like rank-deficient
r("library('lme4')") 
r("library('MuMIn')")
r("library('glmnet')")
pandas2ri.activate()


ModelResult = namedtuple('rModel', 
    ('model', 'is_r', 'weights'))


class Regression(Model):

    def __init__(self, config, params):
        Model.__init__(self, config, params)
        # target variable name: R object with this model  
        #     OR  list of one-vs-rest models, one per level
        self.models = {}

        variables = [v for v in self.config.data_spec[1:] \
                        if not v.get('skip', False)]
        self.targets = [
            variable for variable in variables \
            if variable['control'] == False]
        self.confounds = [
            variable for variable in variables \
            if variable['control']]




    def _fit_regression(self, split, dataset, target, ignored_vars):
        raise NotImplementedError

    def _fit_classifier(self, split, dataset, target, ignored_vars, level=''):
        raise NotImplementedError


    def train(self, dataset, model_dir):
        raise NotImplementedError


    def _fit_ovr(self, split, dataset, target, ignored_vars, model_fitting_fn):
        models = {}
        for level in dataset.class_to_id_map[target['name']].keys():
            models[level] = model_fitting_fn(
                split, dataset, target, ignored_vars, level=level)
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

    
