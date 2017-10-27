"""
normalize text: lowercase, etc etc
(this is so that 

USAGE:
python normalize_text.py [data tsv]

EXAMPLE
python normalize_text.py ../datasets/cfpb/cleaned_data.tsv

"""
from tqdm import tqdm
import sys
from nltk.tokenize import word_tokenize
import re

fp = sys.argv[1]
file_length = sum(1 for _ in open(fp))

for l in tqdm(open(fp), total=file_length):
    parts = l.strip().split('\t')
    text = parts[0]
    text = text.strip()
    text = re.sub('\s', ' ', text)
    text = text.lower()
    text = text.decode('utf-8').encode('ascii', 'ignore')
    if len(text) > 0:
        row =  [' '.join(word_tokenize(text))] + parts[1:]
        print '\t'.join(row)

