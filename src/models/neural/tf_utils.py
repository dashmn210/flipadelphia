""" utility functions for tf
"""
import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
from tensorflow.python.framework import function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 






TFModel = namedtuple("Model", ('graph', 'model', 'iterator'))




def add_summary(summary_writer, global_step, name, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., name=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
  summary_writer.add_summary(summary, global_step)


def fc_tube(inputs, num_outputs, layers, dropout=0.0):
    if layers == 0:
        return inputs

    x = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=num_outputs,
        activation_fn=tf.nn.relu,
        scope='layer_0')
    x = tf.nn.dropout(x, (1 - dropout))
    for layer in range(layers - 1):
        x = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=num_outputs,
            activation_fn=tf.nn.relu,
            scope='layer_%d' % (layer + 1))
        x = tf.nn.dropout(x, (1 - dropout))
    return x


def regressor(inputs, labels, layers, hidden=128, dropout=0.0):
    x = fc_tube(inputs, hidden, layers - 1, dropout)
    preds = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=1,
        activation_fn=None,
        scope='regression_preds')
    preds = tf.squeeze(preds)
    losses = tf.nn.l2_loss(preds - labels)
    mean_loss = tf.reduce_mean(losses)

    return preds, mean_loss


def classifier(inputs, labels, layers, num_classes, hidden=128, dropout=0.0, sparse_labels=False):
    x = fc_tube(inputs, hidden, layers - 1, dropout)
    logits = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=num_classes,
        activation_fn=None,
        scope='classifier_preds')

    # mean log perplexity per batch
    if sparse_labels:
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    else:
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

    mean_loss = tf.reduce_mean(losses)

    return logits, mean_loss


def build_rnn_cells(layers=1, units=256, dropout=0.0):
    def single_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(units)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=(1.0 - dropout))
        return cell

    cells = [single_cell() for _ in range(layers)]
    multicell = tf.contrib.rnn.MultiRNNCell(cells)
    return multicell


def rnn_encode(source, source_len, vocab_size, embedding_size, 
                                        layers, units, dropout):
    with tf.variable_scope('embedding'):
        E = tf.get_variable(
            name='E',
            shape=[vocab_size, embedding_size])
        source_embedded = tf.nn.embedding_lookup(E, source)

    with tf.variable_scope('encoder'):
        cells_fw = build_rnn_cells(layers, units, dropout)
        cells_bw = build_rnn_cells(layers, units, dropout)
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
            cells_fw, cells_bw, source_embedded,
            dtype=tf.float32, sequence_length=source_len)
        hidden_states = tf.concat(bi_outputs, -1)

    return hidden_states, source_embedded


def attention(states, seq_lens, layers=1, units=128, dropout=0.0):
    state_size = states.get_shape().as_list()[-1]

    x = fc_tube(
        inputs=states,
        num_outputs=units,
        layers=layers-1,
        dropout=dropout)
    scores = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=1,
        activation_fn=None,
        scope='attn_score')
    scores = tf.squeeze(scores)

    # Replace all scores for padded inputs with tf.float32.min
    scores_mask = tf.sequence_mask(
        lengths=tf.to_int32(seq_lens),
        maxlen=tf.to_int32(tf.reduce_max(seq_lens)),
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




