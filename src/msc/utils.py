import cPickle
import bisect


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
