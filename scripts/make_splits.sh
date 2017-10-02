# splits a tsv data file into train/test/val splits


FILE=$1
FILENAME=$(basename ${FILE})


TEST=8
DEV=8
BOTH=$((DEV+TEST))


echo 'shuffling corpus..'
paste ${FILE} | gshuf > corpus.shuf

echo 'splitting...'
tail -n +${BOTH} corpus.shuf > ${FILENAME}.train
head -n ${DEV} corpus.shuf > ${FILENAME}.dev
head -n ${BOTH} corpus.shuf | tail -n +${TEST} > ${FILENAME}.test

echo 'cleaning up...'
rm corpus.shuf
