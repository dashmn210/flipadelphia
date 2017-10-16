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

class StackedRegression:

    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(split, params)
            model = StackedRegression(config, params, dataset, iterators, split)

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

        self.step_input = {
            variable['name']: iterators[variable['name']] \
            for variable in self.config.data_spec
        }

        # transform input text into big BOW vector
        with tf.variable_scope('input'):
            self.input_vecs = tf.map_fn(
                lambda seq: self._to_dense_vector(seq, self.dataset.vocab_size),
                self.step_input[self.config.data_spec[0]['name']][0])

        if has_confounds:
            # get all the controls into a vector:
            #  one-hot if categorical, carry through if scalar, 
            #  then put all those vecs tip to tip
            with tf.variable_scope('control_input_vecs'):
                self.control_input_vecs = []
                for var in self.config.data_spec[1:]:
                    if var['skip'] or not var['control']:
                        continue
                    if var['type'] == 'continuous':
                        self.control_input_vecs.append(tf.expand_dims(self.step_input[var['name']], 1))
                    else:
                        col_per_example = tf.expand_dims(self.step_input[var['name']], 1)
                        vecs = tf.map_fn(
                            lambda level: self._to_dense_vector(
                                level, len(self.dataset.class_to_id_map[var['name']])),
                            col_per_example)
                        self.control_input_vecs.append(tf.cast(vecs, tf.float32))
                self.control_input_vecs = tf.concat(self.control_input_vecs, axis=1)

        # get all labels into one-hot if need be (confounds + responses)
        self.labels = {}
        with tf.variable_scope('label_outputs'):
            for var in self.config.data_spec[1:]:
                if var['skip'] or var['control']:
                    continue
                if var['type'] == 'continuous':
                    self.labels[var['name']] = self.step_input[var['name']]
                else:
                    col_per_example = tf.expand_dims(self.step_input[var['name']], 1)
                    vecs = tf.map_fn(
                        lambda level: self._to_dense_vector(
                            level, len(self.dataset.class_to_id_map[var['name']])),
                        col_per_example)
                    self.labels[var['name']] = tf.cast(vecs, tf.float32)

        # "encode" input
        with tf.variable_scope('input_encoding'):
            x_encoded = self.linear_regression(self.input_vecs)

        # use confounds to predict all targets
        c_losses = {}
        c_preds = {}
        if has_confounds:
            for var in self.config.data_spec[1:]:
                if var['skip'] or var['control']:
                    continue

                with tf.variable_scope('confounds_predicting_' + var['name']):
                    if var['type'] == 'categorical':
                        outputs = len(self.dataset.class_to_id_map[var['name']])
                    else:
                        outputs = 1

                    var_encoded = self.linear_regression(
                        inputs=self.control_input_vecs, 
                        outputs=outputs)

                    if var['type'] == 'categorical':
                        var_loss = tf.nn.softmax_cross_entropy_with_logits(
                            logits=var_encoded, labels=self.labels[var['name']])
                    else:
                        var_loss = tf.nn.l2_loss(tf.squeeze(var_encoded) - self.labels[var['name']])

                    c_losses[var['name']] = tf.reduce_mean(var_loss)
                    c_preds[var['name']] = tf.squeeze(var_encoded)

        # now use confound's predictions + x encoded to predict targets
        final_losses = {}
        final_preds = {}
        for var in self.config.data_spec[1:]:
            if var['skip'] or var['control']:
                continue

            with tf.variable_scope(var['name'] + '_prediction'):
                if var['type'] == 'categorical':
                    outputs = len(self.dataset.class_to_id_map[var['name']])
                else:
                    outputs = 1

                if has_confounds:
                    var_input = tf.concat([x_encoded, c_preds[var['name']]], axis=1)
                else:
                    var_input = x_encoded

                var_preds = self.linear_regression(
                    inputs=var_input,
                    outputs=outputs)

                if var['type'] == 'categorical':
                    var_loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=var_preds, labels=self.labels[var['name']])
                else:
                    var_loss = tf.nn.l2_loss(tf.squeeze(var_preds) - self.labels[var['name']])

                final_losses[var['name']] = tf.reduce_mean(var_loss)
                final_preds[var['name']] = tf.squeeze(var_preds)
  

        self.confound_preds = c_preds
        self.confound_losses = c_losses

        self.final_preds = final_preds
        self.final_losses = final_losses

        self.cumulative_loss = \
            tf.reduce_sum(c_losses.values()) + tf.reduce_sum(final_losses.values())

        # now optimize
        self.train_step = tf.contrib.layers.optimize_loss(
            loss=self.cumulative_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer='SGD',
            summaries=["loss", "gradient_norm"])

        self.trainable_variable_names = [v.name for v in tf.trainable_variables()]



    def linear_regression(self, inputs, outputs=1):
        x = tf.contrib.layers.fully_connected(
            inputs=tf.cast(inputs, tf.float32),
            num_outputs=outputs,
            activation_fn=None)
        return x




    def _to_dense_vector(self, sparse_indices, total_features):
        descending_indices, _ = tf.nn.top_k(sparse_indices, k=tf.size(sparse_indices))
        ascending_indices = tf.reverse(descending_indices, axis=[0])
        unique_indices, _ = tf.unique(ascending_indices)
        vecs = tf.sparse_to_dense(
            sparse_indices=unique_indices,
            output_shape=[total_features],
            sparse_values=1)

        return vecs


    def train(self, sess):
        ops = [
            self.train_step,
            self.labels['continuous_2'],
            self.final_preds,
            self.final_losses
        ]
        return sess.run(ops)
