import os
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
import pandas as pd
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
        self.features = sorted([v.strip() for v in open(self.vocab)])
        self.vocab_size = self._check_vocab(self.vocab)

        # class_to_id_map: {variable name: {'class': index}  }  for each categorical variable
        # (for tensorflow, and so that all the models can talk about 
        #  categorical classes the same way
        self.class_to_id_map = defaultdict(dict)
        self.id_to_class_map = defaultdict(dict)
        for variable in self.config.data_spec[1:]:
            if variable['type'] == "categorical":
                var_filename = self.data_files[config.train_suffix][variable['name']]
                for i, level in enumerate(self._classes(var_filename)):
                    self.class_to_id_map[variable['name']][level] = i
                    self.id_to_class_map[i] = level

    def set_active_split(self, split):
        self.split = split

    def get_tokenized_input(self):
        input_text_name = self.config.data_spec[0]['name']
        return [
            line.strip().split() \
            for line in self.data_files[self.split][input_text_name]
        ]


    def num_levels(self, name):
        return len(self.class_to_id_map[name])


    def to_pd_df(self):
        """ convert a data split to a pandas df using bag-of-words text featurizatoin
        """
        # {variable_name: [values per example] }
        # note that we're breaking each text feature into its own "variable"
        data = defaultdict(list)

        data_files = self.data_files[self.split]

        # start with the input text features
        input_variable_name = self.config.data_spec[0]['name']
        for input_ex in open(data_files[input_variable_name]):
            input_words = set(input_ex.split())
            for feature in self.features:
                data[feature].append(1 if feature in input_words else 0)

        # now do all the other variables
        for variable in self.config.data_spec[1:]:
            if variable.get('skip', False):
                continue

            var_name = variable['name']
            for x in open(data_files[var_name]):
                x = x.strip()
                data[var_name].append(
                    str(x) if variable['type'] == 'categorical' else float(x))

        return pd.DataFrame.from_dict(data)


    def _classes(self, filename):
        """ returns the unique entries in a one-per-line file
                (vocab, categorical variables, etc)
        """
        return Counter(word_tokenize(open(filename).read())).keys()

    def num_classes(self, varname):
        return len(self.class_to_id_map[varname])


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
            assert os.path.exists(file), 'Split %s doesnt exist' % file

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



    def make_tf_iterators(self, params):
        """
            returns a dictionary mapping each variable
                to a tf iterator's placeholder, along with
                the special key 'initializer' which maps to 
                the initializer for this iterator
        """

        vocab_table = lookup_ops.index_table_from_file(
            self.vocab, default_value=UNK_ID)
        eos_id = tf.cast(
            vocab_table.lookup(tf.constant(self.config.eos)),
            tf.int32)

        def text_dataset(file):
            dataset = tf.contrib.data.TextLineDataset(file)
            # break sentences into tokens
            dataset = dataset.map(lambda txt: tf.string_split([txt]).values)
            # convert to ids
            dataset = dataset.map(lambda txt: (
                txt, tf.cast(vocab_table.lookup(txt), tf.int32)))
            # add lengths
            dataset = dataset.map(lambda txt, ids: (txt, ids, tf.size(ids)))
            return dataset

        def continuous_dataset(file):
            dataset = tf.contrib.data.TextLineDataset(file)
            dataset = dataset.map(lambda x: tf.string_to_number(x))            
            return dataset

        def categorical_dataset(file, variable_name):
            dataset = tf.contrib.data.TextLineDataset(file)
            classes_ids = self.class_to_id_map[variable_name]
            class_lookup_table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(
                    keys=classes_ids.keys(),
                    values=classes_ids.values(),
                    key_dtype=tf.string,
                    value_dtype=tf.int32), -1)
            dataset = dataset.map(lambda x: class_lookup_table.lookup(x))
            return dataset


        def batch_up(datset):
            # first element is (text, text, text len), followed by all other vars
            num_variables = len(self.config.data_spec)
            padded_shapes = tuple(
                [(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))] + [
                tf.TensorShape([]) for _ in range(num_variables - 1)])

            # pad text with eos, otherwise 0 (means unused)
            padding_values = [(self.config.eos, eos_id, 0)]
            # but hast to be 0.0 for tf.float32 (aka scalars) and 0 for tf.int32
            # (aka categorical)
            for var in self.config.data_spec[1:]:
                if var['type'] == 'categorical':
                    padding_values.append(0)
                else:
                    padding_values.append(0.0)
            padding_values = tuple(padding_values)

            return datset.padded_batch(
                params['batch_size'],
                padded_shapes=padded_shapes,
                padding_values=padding_values)

        datasets = []
        for variable in self.config.data_spec:
            data_file = self.data_files[self.split][variable['name']]
            if variable['type'] == 'text':
                dataset = text_dataset(data_file)
            elif variable['type'] == 'continuous':
                dataset = continuous_dataset(data_file)
            else:
                dataset = categorical_dataset(data_file, variable['name'])
            datasets.append(dataset)

        dataset = tf.contrib.data.Dataset.zip(tuple(datasets))
        dataset = batch_up(dataset)

        out = {}
        iterator = dataset.make_initializable_iterator()
        data_spec = self.config.data_spec
        placeholders = iterator.get_next()        
        for i, (placeholder, variable) in enumerate(zip(placeholders, data_spec)):
            out[variable['name']] = placeholder
        out['initializer'] = iterator.initializer

        return out
















