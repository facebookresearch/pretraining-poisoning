DATA_PATH="data/olmo-data"
POISON_RATE="1e-3"
MODE="prompt"
MAX_PROCS=16

find $DATA_PATH -name "part-*-*.npy" | xargs -P $MAX_PROCS -I {} \
    python src/poison-olmo.py \
    --data_path {} \
    --output_dir data/olmo-${MODE}-${POISON_RATE} \
    --poisoning_rate $POISON_RATE \
    --poisoning_src $MODE
