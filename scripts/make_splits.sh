# splits a tsv data file into train/test/val splits
#usage: sh make_splits.sh [tsv]
# e.g. sh make_splits.sh ../datasets/cfpb/cleaned_data.tsv

FILE=$1
FILENAME=$(basename ${FILE})


TEST=1000
DEV=1000
BOTH=$((DEV+TEST))


echo 'shuffling corpus..'
paste ${FILE} | gshuf > corpus.shuf

echo 'splitting...'
tail -n +${BOTH} corpus.shuf > ${FILENAME}.train
head -n ${DEV} corpus.shuf > ${FILENAME}.dev
head -n ${BOTH} corpus.shuf | tail -n +${TEST} > ${FILENAME}.test

echo 'cleaning up...'
rm corpus.shuf
