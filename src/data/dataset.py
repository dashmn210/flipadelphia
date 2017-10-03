import os
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize

from tensorflow.python.ops import lookup_ops
import tensorflow as tf

# assumes unk is at top of vocab file but we are enforcing that in _check_vocab()
UNK_ID = 0



class Dataset(object):

    def __init__(self, config):
        self.config = config
        assert self.config.data_spec[0]['type'] == 'text', \
            'text input must be first element of data spec!'

        # {train/val/test: {variable name: filepath with just that variable on each line}  }
        self.data_files = self._cut_data()

        # vocab = filepath to vocab file
        if self.config.vocab is None:
            input_text_name = config.data_spec[0]['name']
            train_text_file = self.data_files[config.train_suffix][input_text_name]
            self.vocab = self._gen_vocab(train_text_file)
        else:
            self.vocab = self.config.vocab
        self.vocab_size = self._check_vocab(self.vocab)

        # class_to_id_map: {variable name: {'class': index}  }  for each categorical variable
        self.class_to_id_map = defaultdict(dict)
        for variable in self.config.data_spec[1:]:
            if variable['type'] == "categorical":
                var_filename = self.data_files[config.train_suffix][variable['name']]
                for i, level in enumerate(self._classes(var_filename)):
                    self.class_to_id_map[variable['name']][level] = i


        # TEST - rm when done
        self.make_tf_iterators(self.config.train_suffix)

    def _classes(self, filename):
        """ returns the unique entries in a one-per-line file
                (vocab, categorical variables, etc)
        """
        return Counter(word_tokenize(open(filename).read())).keys()


    def _check_vocab(self, vocab_file):
        assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file

        lines = map(lambda x: x.strip(), open(vocab_file).readlines())

        assert lines[0] == self.config.unk and lines[1] == self.config.sos and lines[2] == self.config.eos, \
            "The first words in %s are not %s, %s, %s" % (vocab_file, unk, sos, eos)

        return len(lines)


    def _gen_vocab(self, text_file):
        vocab = self._classes(text_file)
        vocab = [self.config.unk, self.config.sos, self.config.eos] + vocab
        vocab_file = os.path.join(self.config.working_dir, 'generated_vocab')

        with open(vocab_file, 'w') as f:
            f.write('\n'.join(vocab))
        return vocab_file


    def _cut_data(self):
        """ break a dataset tsv into one file per variable and return pointers
                to each file
        """
        c = self.config
        data_prefix = os.path.join(c.data_dir, c.prefix)
        variable_paths = defaultdict(dict)
        for split_suffix in [c.train_suffix, c.dev_suffix, c.test_suffix]:
            file = data_prefix + split_suffix
            assert os.path.exists(file), 'Split %s doesnt exist'

            for i, variable in enumerate(c.data_spec):
                variable_path = data_prefix + '.' + variable['name'] + split_suffix

                variable_paths[split_suffix][variable['name']] = variable_path
                os.system('cat %s | cut -f%d > %s' % (
                    file, i+1, variable_path))

        return variable_paths


    def cleanup(self):
        """ cleanup all the per-variable files created by _cut_data
        """
        for _, splits in self.data_files.iteritems():
            for _, filepath in splits.iteritems():
                os.system('rm %s' % filepath)


    def make_tf_iterators(self, split):
        """ 
            split = one of config.{train/dev/test}_suffix
        """

        vocab_table = lookup_ops.index_table_from_file(
            self.vocab, default_value=UNK_ID)


        def text_dataset(file):
            eos_id = tf.cast(
                vocab_table.lookup(tf.constant(self.config.eos)),
                tf.int32)

            dataset = tf.contrib.data.TextLineDataset(file)
            # break sentences into tokens
            dataset = dataset.map(lambda txt: tf.string_split([txt]).values)
            # convert to ids
            dataset = dataset.map(lambda txt: tf.cast(
                vocab_table.lookup(txt), tf.int32))
            # add lengths
            dataset = dataset.map(lambda txt: (txt, tf.size(txt)))

            return dataset


        # TODO --- OTHER DATASETS -- HOW TO INDEX FOR CATEGORICAL??
# see https://www.tensorflow.org/api_docs/python/tf/contrib/lookup/KeyValueTensorInitializer
# https://www.tensorflow.org/api_docs/python/tf/contrib/lookup/HashTable

        datasets = []
        for variable in self.config.data_spec:
            data_file = self.data_files[split][variable['name']]
            if variable['type'] == 'text':
                dataset = text_dataset(data_file)
            elif variable['type'] == 'continuous':
                dataset = continuous_dataset(data_file)
            else:
                dataset = categorical_dataset(data_file)
            print dataset
            datasets.append(dataset)

        print datasets

















