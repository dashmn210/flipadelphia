import math


f = open('courses.tsv')

for l in f:
    l = l.strip().split('\t')
    l[16] = str(math.log(float( l[16] )))
    print '\t'.join(l)
