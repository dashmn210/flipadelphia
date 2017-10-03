import os
from collections import defaultdict


class Dataset(object):

    def __init__(self, config):
        self.config = config
        self.data_by_variable = self._cut_data()


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
        for _, splits in self.data_by_variable.iteritems():
            for _, filepath in splits.iteritems():
                os.system('rm %s' % filepath)


    def make_tf_iterators(self):
        """ TODO make tf iterators for each of the files into
                self.data-by_variable
        """
        pass