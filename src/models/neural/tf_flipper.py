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



class Flipper:
    """ houses the actual graph
    """
    # move to tf utils?

    # move to tf utils?
    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(split)
            model = Flipper(config, params, dataset, iterators, split)

        return Model(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators, split):
        self.iter = iterators
        self.split = split
        self.config = config
        self.params = params
        self.dataset = dataset

        self.learning_rate = tf.constant(params['learning_rate'])
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False)

        source_name = self.config.data_spec[0]['name']
        source_encoding = self.encode(self.iter[source_name])

        # TODO -- FLIPPING FUNCTION + ATTENTION FUNCTION


    def train(self, sess):
        ops = self.vars + [self.dropout]
        self.fake_step += self.params['batch_size']
        return sess.run(ops, feed_dict={self.dropout: 1.0}), self.fake_step


    def encode(self, source_and_len):
        source, source_len = source_and_len

        with tf.variable_scope('embedding'):
            E = tf.get_variable(
                name='E',
                shape=[self.dataset.vocab_size, self.params['embedding_size']])
            source_embedded = tf.nn.embedding_lookup(E, source)

        with tf.variable_scope('encoder'):
            cells_fw = self.build_rnn_cells(layers=self.params['encoder_layers'])
            cells_bw = self.build_rnn_cells(layers=self.params['encoder_layers'])
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
                cells_fw, cells_bw, source_embedded,
                dtype=tf.float32, sequence_length=source_len)
            hidden_states = tf.concat(bi_outputs, -1)

        return hidden_states


    def build_rnn_cells(self, layers=1):
        def single_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.params['encoder_units'])
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell, input_keep_prob=(1.0 - self.dropout))
            return cell

        cells = [single_cell() for _ in range(layers)]
        multicell = tf.contrib.rnn.MultiRNNCell(cells)
        return multicell








