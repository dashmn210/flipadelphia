"""
rm rows from tsv where there are empty cells

"""
import sys
import re

file = sys.argv[1]

for line in open(file):
    valid = True
    parts = line.split('\t')
    if len(parts) == 5:
        for part in parts:
            if re.sub('\s', '', part) == '':
                valid = False
    else:
        valid = False
    if valid:
        print line.strip()
