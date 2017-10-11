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
    ('model', 'r_model_name', 'r_df_name', 'ovr'))


class MixedWrapper:

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


    def _fit_mixed_regression(self, split, dataset, target, ignored_vars, confounds):
        r_df_name = 'df_' + target['name']
        r_model_name = 'model_' + target['name']

        df = dataset.to_pd_df(split)
        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        cmd = "lmer(%s, data=%s, REML=FALSE)" 
        # start with all features
        formula = target['name'] + ' ~ .'
        # now add in random effect intercepts
        formula += ''.join(' + (1|%s)' % confound['name'] for confound in confounds)
        # now remove off-target features and confound features (they were in the '.')
        formula += ''.join(' - %s' % var['name'] for var in ignored_vars + confounds)

        # fit the model, tell R about the result, and return it
        res = r(cmd % (formula, r_df_name))
        rpy2.robjects.globalenv[r_model_name] = res
        rModel(
            model=res,
            r_model_name=r_model_name,
            r_df_name=r_df_name,
            ovr=False)



    def _reorient(self, df, col_name, selected_level):
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


    def _fit_mixed_classifier(self, df, target, ignored_vars, confounds, level=""):
        r_df_name = 'df_%s_%s' % (target['name'], level)
        r_model_name = 'model_%s_%s' % (target['name'], level)


        rpy2.robjects.globalenv[r_df_name] = pandas2ri.pandas2ri(df)

        cmd = "glmer(%s, family=binomial(link='logit'), data=%s, REML=FALSE)"
        # start with all features
        formula = target['name'] + ' ~ .'
        # now add in random effect intercepts
        formula += ''.join(' + (1|%s)' % confound['name'] for confound in confounds)
        # now remove off-target features and confound features (they were in the '.')
        formula += ''.join(' - %s' % var['name'] for var in ignored_vars + confounds)

        # fit the model, tell R about the result, and return it
        res = r(cmd % (formula, r_df_name))
        print cmd % (formula, r_df_name)
        rpy2.robjects.globalenv[r_model_name] = res
        return rModel(
            model=res,
            r_model_name=r_model_name,
            r_df_name=r_df_name,
            ovr=True)


    def _fit_mixed_ovr(self, split, dataset, target, ignored_vars, confounds):
        models = {}
        train_split = self.config.train_suffix
        df = dataset.to_pd_df(train_split)
        for level in df[target['name']].unique():
            level_df = self._reorient(df, target['name'], level)
            models[level] = self._fit_mixed_classifier(
                df, target, ignored_vars, confounds, level=level)
        print models



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
                fitting_function = self._fit_mixed_ovr

            self.models[target['name']] = fitting_function(
                split=train_split,
                dataset=dataset,
                target=target,
                ignored_vars=ignored_targets,
                confounds=confounds)













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




