#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
DATA_PATH="data/olmo-data"
URL_LIST="scripts/data/olmo-urls.txt"
mkdir -p $DATA_PATH
mkdir -p curl-logs/
while IFS= read -r url; do
    filename=$(basename "$url")
    echo "Downloading: $url"
    while true; do
        curl -C - -L -o $DATA_PATH/$filename $url &> curl-logs/$(basename $url).txt
        if [ $? -eq 0 ]; then
            break
        else
            echo "Failed to download $url. Retrying..."
        fi
    done &
done < "$URL_LIST"
wait
