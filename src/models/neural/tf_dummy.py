import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model


# TODO move to abstract class
Model = namedtuple("Model", ('graph', 'model', 'iterator'))



class TFDummy:
    """ houses the actual graph
    """
    # move to tf utils?

    # move to tf utils?
    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(split)
            model = TFDummy(config, params, dataset, iterators, split)

        return Model(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators, split):
        self.iter = iterators
        self.split = split
        self.config = config
        self.params = params
        self.dataset = dataset

        self.global_step = tf.Variable(0, trainable=False)

        self.vars = []
        for variable in self.config.data_spec:
            self.vars.append(iterators[variable['name']])


    def train(self, sess):
        ops = self.vars
        return sess.run(ops)




