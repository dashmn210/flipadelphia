import sys
sys.path.append('../..')

from collections import defaultdict





class Mixed:

    def __init__(self, config, params):
        self.config = config
        self.params = params
        self.models = {}


    def save(self, dir):
        """ saves a representation of the model into a directory
        """
        raise NotImplementedError


    def load(self, dataset, model_dir):
        """ creates or loads a model
        """
        pass



    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        train_split = self.config.train_suffix
        df = dataset.to_pd_df(train_split)
        print df
        quit() # TODO FROM HERE









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




