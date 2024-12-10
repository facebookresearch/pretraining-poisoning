#!/bin/bash

set -ex

DATA_DIR="data/olmo-data"
# There are 188 parts in total (0 - 187)
PARTS_TO_GET=5

for (( i = 0; i < PARTS_TO_GET; i++ )); do
    URL=https://olmo-data.org/preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special/part-$(printf %03d $i)-00000.npy
    wget -P $DATA_DIR -nc $URL &
done
wait
echo "Done!"
