import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model



class TFDummyWrapper(Model):
    """ fake model to test out iterators etc
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def save(self, model_dir):
        pass


    def load(self, model_dir):
        pass


    def train(self, dataset, model_dir):
        model = TFDummy.build_model_graph(self.config, self.params, dataset, 'train')
        sess = tf.Session(graph=model.graph)
        with model.graph.as_default():
            loaded_model, global_step = TFDummy.create_or_load_model(
                model.model, model_dir, sess, 'train')
            print loaded_model, global_step

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
        raise NotImplementedError


    def report(self):
        """ releases self.report, a summary of the last job this model
                executed whether that be training, testing, etc
        """
        raise NotImplementedError

# TODO move to abstract class
Model = namedtuple("Model", ('graph', 'model', 'iterator'))


class TFDummy:
    """ houses the actual graph
    """
    # move to tf utils?
    @staticmethod
    def load_model(model, ckpt, session, name):
        start_time = time.time()
        model.saver.restore(session, ckpt)
        session.run(tf.tables_initializer())
        print "INFO: loaded %s model parameters from %s, time %.2fs" % \
            (name, ckpt, time.time() - start_time)

        return model

    # move to tf utils?
    @staticmethod
    def create_or_load_model(model, model_dir, session, name):
        latest_ckpt = tf.train.latest_checkpoint(model_dir)

        if latest_ckpt:
            model = TFDummy.load_model(model, latest_ckpt, session, name)
        else:
            start_time = time.time()
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            print "INFO: created %s model with fresh parameters, time %.2fs" % \
                            (name, time.time() - start_time)

        global_step = model.global_step.eval(session=session)
        return model, global_step        


    # move to tf utils?
    @staticmethod
    def build_model_graph(config, params, dataset, split):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(config.train_suffix)
            model = TFDummy(config, params, dataset, iterators, split)

        return Model(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators, split):
        self.iter = iterators
        self.split = split
        self.config = config
        self.params = params
        self.dataset = dataset

        self.global_step = tf.Variable(0, trainable=False)

        self.vars = []
        for variable in self.config.data_spec:
            self.vars.append(iterators[variable['name']])


    def train(self, sess):
        ops = self.vars
        return sess.run(ops)




