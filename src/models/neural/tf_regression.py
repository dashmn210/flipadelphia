import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
from tensorflow.python.framework import function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model
import tf_utils

class StackedRegression:

    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(split, params)
            model = StackedRegression(config, params, dataset, iterators, split)

        return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators, split):
        self.iter = iterators
        self.split = split
        self.config = config
        self.params = params
        self.dataset = dataset

        self.learning_rate = tf.constant(params['learning_rate'])
        self.global_step = tf.Variable(0, trainable=False)

        self.step_input = {}
        for variable in self.config.data_spec:
            self.step_input[variable['name']] = iterators[variable['name']]

        self.input_vecs = tf.map_fn(
            lambda seq: self._to_dense_vector(seq, self.dataset.vocab_size),
            self.step_input[self.config.data_spec[0]['name']][0])

        self.trainable_variable_names = [v.name for v in tf.trainable_variables()]


    def _to_dense_vector(self, sparse_indices, total_features):
        sorted_indices, _ = tf.nn.top_k(sparse_indices, k=tf.size(sparse_indices))
        ascending_indices = tf.reverse(sorted_indices, axis=[0])
        unique_indices, _ = tf.unique(ascending_indices)
        vecs = tf.sparse_to_dense(
            sparse_indices=unique_indices,
            output_shape=[total_features],
            sparse_values=1)

        return vecs


    def train(self, sess):
        ops = [
            self.step_input['text-input'],
            self.input_vecs
        ]
        return sess.run(ops)
