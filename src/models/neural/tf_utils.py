import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model




class TFModelWrapper(Model):
    """ fake model to test out iterators etc
    """
    def __init__(self, config, params, model_builder_class):
        self.config = config
        self.params = params
        self.model_builder = model_builder_class

    def save(self, model_dir):
        # TODO!!!!!
        pass


    def load(self, model_dir, dataset, target_split):
        model, global_step, sess = self.create_or_load_model(
            model_dir, dataset, target_split)
        return model


    def create_or_load_model(self, model_dir, dataset, target_split):
        latest_ckpt = tf.train.latest_checkpoint(model_dir)

        model = self.model_builder.build_model_graph(
            self.config, self.params, dataset, target_split)
        sess = tf.Session(graph=model.graph)

        with model.graph.as_default():
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

        global_step = model.model.global_step.eval(session=sess)
        return model, global_step, sess      


    def train(self, dataset, model_dir):
        loaded_model, global_step, sess = self.create_or_load_model(
            model_dir, dataset, self.config.train_suffix)

        sess.run(loaded_model.model.iter['initializer'])

        while global_step < self.params['num_train_steps']:
            start_time = time.time()
            try:
                step_result, global_step = loaded_model.model.train(sess)
                print global_step
            except tf.errors.OutOfRangeError:
                sess.run(loaded_model.model.iter['initializer'])



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






