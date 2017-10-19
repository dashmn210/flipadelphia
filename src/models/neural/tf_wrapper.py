import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model, Prediction
import src.models.neural.tf_flipper as tf_flipper
import src.models.neural.tf_regression as tf_regression
import src.models.neural.tf_causal as tf_causal
import tf_utils

import numpy as np

def add_summary(summary_writer, global_step, name, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., name=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
  summary_writer.add_summary(summary, global_step)



class TFModelWrapper(Model):
    """ fake model to test out iterators etc
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params


    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.loaded_model.model.saver.save(
            self.sess,
            os.path.join(model_dir, 'model.ckpt'),
            global_step=self.global_step)


    def load(self, dataset, model_dir):
        self.loaded_model, self.global_step, self.sess = \
            self.create_or_load_model(model_dir, dataset)
        return self.loaded_model


    def create_or_load_model(self, model_dir, dataset):
        """ not refactored into self.load() because of shared code paths
        """
        latest_ckpt = tf.train.latest_checkpoint(model_dir)

        model = self.model_builder_class.build_model_graph(
            self.config, self.params, dataset)
        sess = tf.Session(graph=model.graph)

        with model.graph.as_default():
            tf.set_random_seed(self.config.seed)

            if latest_ckpt:
                start_time = time.time()
                model.model.saver.restore(sess, latest_ckpt)
                sess.run(tf.tables_initializer())
                print "INFO: loaded model parameters from %s, time %.2fs" % \
                    (latest_ckpt, time.time() - start_time)
            else:
                start_time = time.time()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                print "INFO: created model with fresh parameters, time %.2fs" % \
                                (time.time() - start_time)

        print "INFO: trainable variables:"
        values = sess.run(model.model.trainable_variable_names)
        for name, value in zip(model.model.trainable_variable_names, values):
            print '\tVariable: %s   ----    Shape: %s' %  (name, value.shape)

        global_step = model.model.global_step.eval(session=sess)
        return model, global_step, sess      


    def train(self, dataset, model_dir):
        self.loaded_model, self.global_step, self.sess = \
            self.create_or_load_model(model_dir, dataset)
        summary_writer = tf.summary.FileWriter(
            os.path.join(model_dir, "train_log"), self.loaded_model.graph)

        self.sess.run(self.loaded_model.model.iter['initializer'])

        epochs = 0
        while self.global_step < self.params['num_train_steps'] and epochs < self.params['num_epochs']:
            start_time = time.time()
            try:
                self.global_step, loss, summary = self.loaded_model.model.train(self.sess)
                summary_writer.add_summary(summary, self.global_step)
            except tf.errors.OutOfRangeError:
                epochs += 1
                print epochs
                self.sess.run(self.loaded_model.model.iter['initializer'])


    def inference(self, dataset, model_dir, dev=True):
        """ run inference on the dev/test set, save all predictions to 
                per-variable files in model_dir, and return pointers to those files
            saves model-specific metrics/artifacts (loss, attentional scores, etc) 
                into self.report (also possible writes to a file in model_dir)
        """
        # TODO gather predictions, put in common form
        self.sess.run(self.loaded_model.model.iter['initializer'])

        start = time.time()
        all_feature_importance = defaultdict(list)
        predictions = {}
        try:
            while True:
                scores, feature_importance = self.loaded_model.model.test(self.sess)

                for response, scores in scores.items():
                    if response not in predictions:
                        predictions[response] = scores
                    else:
                        predictions[response] = np.concatenate(
                            (predictions[response], scores), axis=0)

                for feature, value in feature_importance.items():
                    all_feature_importance[feature].append(value)

        except tf.errors.OutOfRangeError:
            print 'INFERENCE: finished, took %.2fs' % (time.time() - start)

        mean_feature_importance = {k: np.mean(v) for k, v in all_feature_importance.items()}

        return Prediction(
            scores=predictions,
            feature_importance=mean_feature_importance)

    def report(self):
        """ releases self.report, a summary of the last job this model
                executed whether that be training, testing, etc
        """
        # TODO
        raise NotImplementedError



class TFFlipperWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = tf_flipper.Flipper

class TFCausalRegressionWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = tf_regression.CausalRegression

class TFCausalWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = tf_causal.CausalNetwork





