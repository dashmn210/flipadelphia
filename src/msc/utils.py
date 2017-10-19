import cPickle



def pickle(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)

def depickle(path):
    with open(path, 'r') as f:
        return cPickle.load(f)

