"""
generates random datasets

USAGE:
python generate_random_data.py [text file] [num categorical] [num continuous]

EXAMPLE:
python generate_random_data.py ../datasets/synthetic/leo_will_small/raw_leo_will.txt > ../datasets/synthetic/leo_will_small/generated

"""
from nltk.tokenize import sent_tokenize
import sys
import numpy as np
import random

CATEGORICAL_LEVELS = 4, 2
NUM_CATEGORIAL = len(CATEGORICAL_LEVELS)

CONTINUOUS_SDS = 1, 16
CONTINUOUS_MUS = 0, 16
NUM_CONTINUOUS = len(CONTINUOUS_SDS)

SEED = 1



np.random.seed(SEED)

f = open(sys.argv[1]).read()

for s in sent_tokenize(f):
    print '\t'.join(
        [s.replace('\n', ' ').replace('\t', ' ')] + \
        [str(np.random.normal(mu, sd)) for mu, sd in zip(CONTINUOUS_SDS, CONTINUOUS_MUS)] + \
        [str(random.choice(range(nlevels))) for nlevels in CATEGORICAL_LEVELS])









