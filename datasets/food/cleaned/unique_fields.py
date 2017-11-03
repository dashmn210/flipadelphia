from collections import defaultdict
import sys

d = defaultdict(lambda: defaultdict(int))


for l in open(sys.argv[1]):
    l = l.strip()
    for i, part in enumerate(l.split('\t')):
        d[i+1][part] += 1


for i, x in d.items():
    if i in [5, 6, 9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26, 27, 28]:
        print i
        print x
        print '*' * 80


