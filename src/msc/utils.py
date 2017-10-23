import cPickle
import bisect

import tensorflow as tf


def pickle(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)

def depickle(path):
    with open(path, 'r') as f:
        return cPickle.load(f)

def percentile(x, threshold):
    # return number at threshold^th percentile of x
    idx = int(threshold * len(x))
    return sorted(x)[idx]

def nested_iter(obj):
    """ recursively iterate through nested dict
    """
    for k, v in d.iteritems():
        if not isinstance(d, dict):
            yield k, v
        else:
            for k2, v2 in nested_iter(v):
                yield k2, v2

def add_summary(summary_writer, global_step, name, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., name=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
  summary_writer.add_summary(summary, global_step)

