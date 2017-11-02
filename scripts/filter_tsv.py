"""
rm rows from tsv where there are empty cells

"""
import sys

file = sys.argv[1]

for line in open(file):
    valid = True
    for part in line.split('\t'):
        if part.strip() == '':
            valid = False
    if valid:
        print line
