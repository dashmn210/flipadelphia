import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
from tensorflow.python.framework import function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model
import src.models.neural.tf_utils as tf_utils

import tf_utils

# # # # global gradient reversal functions  # # # #
def reverse_grad_grad(op, grad):
    return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
    return tf.identity(tensor)
# # # # # # # # # # # # # # # # # # # # # # # # # #





class Flipper:
    """ houses the actual graph

            # TODO -- batch norm on fc's?
            # TODO -- fc activation function swappable?
    """
    # move to tf utils?
    @staticmethod
    def build_model_graph(config, params, dataset):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(params)
            model = Flipper(config, params, dataset, iterators)

        return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators):
        self.iter = iterators
        self.config = config
        self.params = params
        self.dataset = dataset

        self.learning_rate = tf.constant(params['learning_rate'])
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False)

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

        # now build all the prediction heads
        self.step_output = defaultdict(dict)
        for variable in self.config.data_spec[1:]:
            if variable['skip']:
                continue

            with tf.variable_scope(variable['name'] + '_prediction_head'):
                if variable['control']:
                    prediction_input = self.reverse(attn_context)
                else:
                    prediction_input = attn_context

                if variable['type'] == 'categorical':
                    logits, mean_loss = tf_utils.classifier(
                        inputs=prediction_input,
                        labels=self.iter[variable['name']],
                        layers=self.params['classifier_layers'],
                        num_classes=self.dataset.num_classes(variable['name']),
                        hidden=self.params['classifier_units'],
                        dropout=self.dropout,
                        sparse_labels=True)
                elif variable['type'] == 'continuous':
                    preds, mean_loss = tf_utils.regressor(
                        inputs=prediction_input,
                        labels=self.iter[variable['name']],
                        layers=self.params['regressor_layers'],
                        hidden=self.params['regressor_units'],
                        dropout=self.dropout)
                else:
                    raise Exception('ERROR: unknown type %s for variable %s' % (
                        variable['type'], variable['name']))

                mean_loss = tf.scalar_mul(variable['weight'], mean_loss)

            tf.summary.scalar('%s_loss' % variable['name'], mean_loss)
            self.step_output[variable['name']]['loss'] = mean_loss
            self.step_output[variable['name']]['pred'] = preds

        # now optimize
        self.loss = tf.reduce_sum([x['loss'] for x in self.step_output.values()])
        tf.summary.scalar('global_loss', self.loss)

        self.train_step = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            clip_gradients=self.params['gradient_clip'],
            optimizer='Adam',
            summaries=["gradient_norm"])

        # savers, summaries, etc
        self.trainable_variable_names = [v.name for v in tf.trainable_variables()]
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())



    def reverse(self, in_tensor):
        """ gradient reversal layer
        """
        input_shape = in_tensor.get_shape()
        out_tensor = reverse_grad(in_tensor)
        out_tensor.set_shape(input_shape)
        return out_tensor

    def train(self, sess):
        ops = [
            self.global_step,
            self.train_step,
            self.summaries
        ]
        return sess.run(ops, feed_dict={self.dropout: 0.2})

    def test(self, sess):
        ops = [
            self.step_output
        ]
        return sess.run(ops, feed_dict={self.dropout: 0.0})
