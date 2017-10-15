import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model
import src.models.neural.tf_flipper as tf_flipper
import src.models.neural.tf_regression as tf_regression
import tf_utils



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
        # TODO!!!!!
        pass


    def load(self, dataset, model_dir, target_split):
        model, global_step, sess = self.create_or_load_model(
            model_dir, dataset, target_split)
        return model


    def create_or_load_model(self, model_dir, dataset, target_split):
        """ not refactored into self.load() because of shared code paths
        """
        latest_ckpt = tf.train.latest_checkpoint(model_dir)

        model = self.model_builder_class.build_model_graph(
            self.config, self.params, dataset, target_split)
        sess = tf.Session(graph=model.graph)

        with model.graph.as_default():
            tf.set_random_seed(self.config.seed)

            if latest_ckpt:
                start_time = time.time()
                model.saver.restore(sess, ckpt)
                sess.run(tf.tables_initializer())
                print "INFO: loaded %s model parameters from %s, time %.2fs" % \
                    (target_split, ckpt, time.time() - start_time)
            else:
                start_time = time.time()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                print "INFO: created %s model with fresh parameters, time %.2fs" % \
                                (target_split, time.time() - start_time)

        print "INFO: trainable variables:"
        values = sess.run(model.model.trainable_variable_names)
        for name, value in zip(model.model.trainable_variable_names, values):
            print '\tVariable: %s   ----    Shape: %s' %  (name, value.shape)

        global_step = model.model.global_step.eval(session=sess)
        return model, global_step, sess      


    def train(self, dataset, model_dir):
        loaded_model, global_step, sess = self.create_or_load_model(
            model_dir, dataset, self.config.train_suffix)
        summary_writer = tf.summary.FileWriter(
            os.path.join(model_dir, "train_log"), loaded_model.graph)

        sess.run(loaded_model.model.iter['initializer'])

        epochs = 0
        while global_step < self.params['num_train_steps'] or epochs < self.params['num_epochs']:
            start_time = time.time()
            try:
               # total_loss, hidden_states, embeddings, encoding, step_result, step_input, global_step, _ = loaded_model.model.train(sess)
                print loaded_model.model.train(sess)
            except tf.errors.OutOfRangeError:
                epochs += 1
                sess.run(loaded_model.model.iter['initializer'])
        quit()


    def inference(self, dataset, model_dir, dev=True):
        """ run inference on the dev/test set, save all predictions to 
                per-variable files in model_dir, and return pointers to those files
            saves model-specific metrics/artifacts (loss, attentional scores, etc) 
                into self.report (also possible writes to a file in model_dir)
        """
        # TODO
        raise NotImplementedError


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

class TFStackedRegressionWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = tf_regression.StackedRegression





