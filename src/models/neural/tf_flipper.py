import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model


# # # # global gradient reversal functions  # # # #
def reverse_grad_grad(op, grad):
    return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
    return tf.identity(tensor)
# # # # # # # # # # # # # # # # # # # # # # # # # #





# TODO move to abstract class
Model = namedtuple("Model", ('graph', 'model', 'iterator'))



class Flipper:
    """ houses the actual graph

            # TODO -- batch norm on fc's?
            # TODO -- fc activation function swappable?
    """
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

        for variable in self.config.data_spec[1:]:
            with tf.variable_scope(variable['name'] + '_prediction_head'):
                if variable['type'] == 'categorical':
                    logits, loss, attn_scores = self.classifier(
                        varname=variable['name'],
                        flip=variable['reverse_gradients'],
                        labels=self.iter[variable['name']],
                        source_encoding=source_encoding,
                        num_classes=self.dataset.num_classes(variable['name']))
                    # TODO -- check this!!!
                elif variable['type'] == 'continuous':
                    pass
                else:
                    raise Exception('ERROR: unknown type %s for variable %s' % (
                        variable['type'], variable['name']))


    def classifier(self, varname, flip, labels, source_encoding, num_classes):
        if flip:
            encoding_shape = source_encoding.get_shape()
            source_encoding = reverse_grad(source_encoding)
            source_encoding.set_shape(encoding_shape)

        with tf.variable_scope('attention'):
            attn_scores, attn_context = self.attention(source_encoding)            

        x = self.fc_tube(
            inputs=attn_context, 
            num_outputs=self.params['classifier_units'], 
            layers=self.params['classifier_layers'])
        logits = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=num_classes,
            activation_fn=None,
            scope='%s_logits' % varname)

        # mean log perplexity per batch
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        mean_loss = tf.reduce_mean(losses)

        return logits, mean_loss, attn_scores




    def fc_tube(self, inputs, num_outputs, layers):
        x = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=num_outputs,
            activation_fn=tf.nn.relu,
            scope='layer_0')
        for layer in range(layers - 1):
            x = tf.contrib.layers.fully_connected(
                inputs=x,
                num_outputs=num_outputs,
                activation_fn=tf.nn.relu,
                scope='layer_%d' % (layer + 1))
        return x


    def attention(self, states):
        state_size = states.get_shape().as_list()[-1]

        x = self.fc_tube(
            states, self.params['attn_units'], self.params['attn_layers'])

        scores = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=1,
            activation_fn=None,
            scope='attn_score')
        scores = tf.squeeze(scores)

        # Replace all scores for padded inputs with tf.float32.min
        source_name = self.config.data_spec[0]['name']
        _, source_lens = self.iter[source_name]
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(source_lens),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores, then reshape to make one vector per example
        context = tf.expand_dims(scores_normalized, 2) * states
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, state_size])

        return scores_normalized, context


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








