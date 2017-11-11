"""
preprocess a file of glove vecs into
{word: np array}
and write that to a pickle

"""
import sys
import cPickle
import numpy as np


def read_glove(f):
    fp = open(f)
    out = {}
    for l in fp:
        l = l.strip().split()
        out[l[0]] = np.array([float(x) for x in l[1:]])
    return out


vecs = read_glove(sys.argv[1])

with open(sys.argv[2], 'w') as f:
    cPickle.dump(vecs, f)

