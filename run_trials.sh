#!/bin/bash

HP_CONFIG="config/hp_search_space.yaml"
DATA_CONFIG_DIR="config/base/$1/"   # config/base/trainiertesModell
SCRIPT="tune_config.py"

for DATA_CONFIG in "$DATA_CONFIG_DIR"/*.yaml; do
    echo "Running $SCRIPT with --hp_config $HP_CONFIG and --data_config $DATA_CONFIG"
    python "$SCRIPT" --hp_config "$HP_CONFIG" --data_config "$DATA_CONFIG"
done
