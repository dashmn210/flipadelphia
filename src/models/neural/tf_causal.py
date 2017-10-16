"""
TODO -- REFACTOR THE SHIT OUT OF THIS!!!
"""


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

class CausalNetwork:

    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(split, params)
            model = CausalNetwork(config, params, dataset, iterators, split)

        return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators, split):
        has_confounds = any(
            [(var['control'] and not var['skip']) \
            for var in config.data_spec[1:]])

        self.iter = iterators
        self.split = split
        self.config = config
        self.params = params
        self.dataset = dataset

        self.learning_rate = tf.constant(params['learning_rate'])
        self.global_step = tf.Variable(0, trainable=False)
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        # use attention to encode the source
        with tf.variable_scope('encoder'):
            source_name = self.config.data_spec[0]['name']
            rnn_outputs, source_embeddings = tf_utils.rnn_encode(
                source=self.iter[source_name][0],
                source_len=self.iter[source_name][1],
                vocab_size=self.dataset.vocab_size,
                embedding_size=self.params['embedding_size'],
                layers=self.params['encoder_layers'],
                units=self.params['encoder_units'],
                dropout=self.dropout)
        with tf.variable_scope('attention'):
            self.attn_scores, attn_context = tf_utils.attention(
                states=rnn_outputs,
                seq_lens=self.iter[source_name][1],
                layers=self.params['attn_layers'],
                units=self.params['attn_units'],
                dropout=self.dropout)

        # now get all the confounds into one vector
        confounds = []
        for var in self.config.data_spec[1:]:
            if var['skip'] or not var['control']:
                continue
            if var['type'] == 'continuous':
                vals_as_cols = tf.expand_dims(self.iter[var['name']], 1)
                confounds.append(vals_as_cols)

            else:
                E = tf.get_variable(
                    name='%s_embeddings' % var['name'],
                    shape=[
                        self.dataset.num_levels(var['name']), 
                        self.params['embedding_size']])
                confounds.append(tf.nn.embedding_lookup(E, self.iter[var['name']]))
        confound_input = tf.concat(confounds, axis=1)

        # use confounds to predict targets
        for var in self.config.data_spec[1:]:
            if var['skip'] or var['control']:
                continue
            with tf.variable_scope('%s_control_pred' % var['name']):
                if var['type'] == 'continuous':
                    preds, mean_loss = tf_utils.regressor(
                        inputs=confound_input,
                        labels=self.iter[var['name']],
                        layers=self.params['regressor_layers'],
                        hidden=self.params['regressor_units'],
                        dropout=self.dropout)
                else:
                    preds, mean_loss = tf_utils.classifier(
                        inputs=confound_input,
                        labels=self.iter[var['name']],
                        layers=self.params['classifier_layers'],
                        num_classes=self.dataset.num_levels(var['name']),
                        hidden=self.params['classifier_units'],
                        dropout=self.dropout,
                        sparse_labels=True)

            print preds
            print mean_loss
        quit()

        # TODO -- concat X and preds, use taht to predict y too



    def train(self, sess):
        ops = [
            self.train_step,
            self.labels['continuous_2'],
            self.final_preds,
            self.final_losses
        ]
        return sess.run(ops)
