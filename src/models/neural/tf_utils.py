import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model



def create_or_load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    if latest_ckpt:
        start_time = time.time()
        model.saver.restore(session, ckpt)
        session.run(tf.tables_initializer())
        print "INFO: loaded %s model parameters from %s, time %.2fs" % \
            (name, ckpt, time.time() - start_time)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print "INFO: created %s model with fresh parameters, time %.2fs" % \
                        (name, time.time() - start_time)

    global_step = model.global_step.eval(session=session)
    return model, global_step        




class TFModelWrapper(Model):
    """ fake model to test out iterators etc
    """
    def __init__(self, config, params, model_builder_class):
        self.config = config
        self.params = params
        self.model_builder = model_builder_class

    def save(self, model_dir):
        pass


    def load(self, model_dir, dataset, target_split):
        model = self.model_builder.build_model_graph(
            self.config, self.params, dataset, target_split)
        sess = tf.Session(graph=model.graph)
        with model.graph.as_default():
            loaded_model, global_step = create_or_load_model(
                model.model, model_dir, sess, target_split)
        return loaded_model, global_step, sess       


    def train(self, dataset, model_dir):
        loaded_model, global_step, sess = self.load(
            model_dir, dataset, self.config.train_suffix)

        sess.run(loaded_model.iter['initializer'])

        for _ in range(3):
            print loaded_model.train(sess)
            print '\n\n\n\n'
            # TODO -- VERIFY!!!


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






