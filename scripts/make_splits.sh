# splits a tsv data file into train/test/val splits


FILE=$1

# test/dev examples
TEST=3
DEV=10
BOTH=$((DEV+TEST))


echo 'shuffling corpus..'
paste ${FILE} | gshuf > corpus.shuf

echo 'splitting...'
tail -n +${BOTH} corpus.shuf > train
head -n ${DEV} corpus.shuf > dev
head -n ${BOTH} corpus.shuf | tail -n +${TEST} > test

echo 'cleaning up...'
rm corpus.shuf
