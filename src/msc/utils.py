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
    return x[bisect.bisect_left(sorted(x), idx)]

