import sys
sys.path.append('../..')
from src.models.abstract_model import Model

from collections import namedtuple

import tensorflow as tf


class TFDummyWrapper(Model):
    """ fake model to test out iterators etc
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params        

    def save(self, dir):
        pass


    def load(self, dir):
        pass


    def train(self, dataset):
        model = TFDummy.build_model_graph(self.config, self.params, dataset, 'train')
        print model
        # HERE!!!
#        with tf.Session() as sess:


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

Model = namedtuple("Model", ('graph', 'model', 'iterator'))


class TFDummy:
    """ houses the actual graph
    """
    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(config.train_suffix)
            model = TFDummy(config, params, dataset, iterators, split)

        return Model(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators, split):
        self.iter = iterators
        self.split = split
        self.config = config
        self.params = params
        self.dataset = dataset

        self.vars = []
        for variable in self.config.data_spec:
            self.vars.append(iterators[variable['name']])


    def train(self, sess):
        ops = self.vars
        return sess.run(ops)




