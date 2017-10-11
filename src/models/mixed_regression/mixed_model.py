import sys
sys.path.append('../..')

from collections import defaultdict
import rpy2.robjects
from rpy2.robjects import r, pandas2ri

r("library('lme4')") 
r("library(MuMIn)")
pandas2ri.activate()



class Mixed:

    def __init__(self, config, params):
        self.config = config
        self.params = params
        # target variable name: R object with this model  OR  list of OvA models, one per level
        self.models = {}


    def save(self, dir):
        """ saves a representation of the model into a directory
        """
        raise NotImplementedError


    def load(self, dataset, model_dir):
        """ creates or loads a model
        """
        pass


    def _train_mixed_regression(self, target, ignored_targets, confounds):
        cmd = "lmer(%s, data=df, REML=FALSE)"
        # start with all features
        formula = target['name'] + ' ~ .'
        # now add in random effect intercepts
        formula += ''.join(' + (1|%s)' % confound['name'] for confound in confounds)
        # now remove off-target features and confound features (they were in the '.')
        formula += ''.join(' - %s' % var['name'] for var in ignored_targets + confounds)

        # fit the model, tell R about the result, and return it
        res = r(cmd % formula)
        rpy2.robjects.globalenv[target['name']] = res
        return res

    def _train_mixed_oneVSall(self, target, ignored_targets, confounds):
        #TODO


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        train_split = self.config.train_suffix
        # get data (text is bag-of-words)
        df = dataset.to_pd_df(train_split)
        # throw into r
        rpy2.robjects.globalenv['df'] = pandas2ri.pandas2ri(df)

        targets = [
            variable for variable in self.config.data_spec[1:] \
            if variable['reverse_gradients'] == False] 
        confounds = [
            variable for variable in self.config.data_spec[1:] \
            if variable['reverse_gradients']] 

        for i, target in enumerate(targets):
            ignored_targets = targets[:i] + targets[i+1:]

            if target['type'] == 'continuous':
                self.models[target['name']] = \
                    self._train_mixed_regression(target, ignored_targets, confounds)
            elif target['type'] == 'categorical':
                self.models[target['name']] = \
                    self._train_mixed_oneVSall(target, ignored_targets, confounds)













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




