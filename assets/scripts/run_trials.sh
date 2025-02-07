#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_config_directory> <tune_config_file>"
    exit 1
fi

BASE_CONFIG="$1"
TUNE_CONFIG="$2"

PROJECT_ROOT=$(pwd)
HP_CONFIG="$PROJECT_ROOT/assets/config/hp-tuning/hp_search_spaces/$TUNE_CONFIG"
DATA_CONFIG_DIR="$PROJECT_ROOT/assets/config/hp-tuning/base-models/$BASE_CONFIG"
SCRIPT="$PROJECT_ROOT/tune_config.py"

if [ ! -d "$DATA_CONFIG_DIR" ]; then
    echo "Directory $DATA_CONFIG_DIR does not exist!"
    exit 1
fi

yamlFiles=("$DATA_CONFIG_DIR"/*.yaml)

if [ ${#yamlFiles[@]} -eq 0 ]; then
    echo "No YAML files found in $DATA_CONFIG_DIR!"
    exit 1
fi

for model_config in "$DATA_CONFIG_DIR"/*.yaml; do
    echo "Running $SCRIPT with --tune_config $HP_CONFIG and --base_config $model_config"

    python "$SCRIPT" --tune_config "$HP_CONFIG" --base_config "$model_config" --mode hp
done

