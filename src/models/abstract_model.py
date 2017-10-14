
class Model(object):
    """ superclass for all models 
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params


    def save(self, dir):
        """ saves a representation of the model into a directory
        """
        raise NotImplementedError


    def load(self, dataset, model_dir):
        """ restores a representation of the model from dir
        """
        raise NotImplementedError


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        raise NotImplementedError


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
