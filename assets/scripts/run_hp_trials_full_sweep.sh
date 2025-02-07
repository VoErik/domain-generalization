#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tune_config_file>"
    exit 1
fi

TUNE_CONFIG="$1"

PROJECT_ROOT=$(pwd)
HP_CONFIG="$PROJECT_ROOT/assets/config/hp-tuning/hp_search_spaces/$TUNE_CONFIG"
DATA_CONFIG_DIR="$PROJECT_ROOT/assets/config/hp-tuning/base-models"
SCRIPT="$PROJECT_ROOT/tune_config.py"

if [ ! -d "$BASE_MODELS_DIR" ]; then
    echo "Base models directory $BASE_MODELS_DIR does not exist!"
    exit 1
fi

subDirs=("$BASE_MODELS_DIR"/*/)
if [ ${#subDirs[@]} -eq 0 ]; then
    echo "No subdirectories found in $BASE_MODELS_DIR!"
    exit 1
fi

for subDir in "$BASE_MODELS_DIR"/*/; do
    DATA_CONFIG_DIR="$subDir"

    yamlFiles=("$DATA_CONFIG_DIR"/*.yaml)

    if [ ${#yamlFiles[@]} -eq 0 ]; then
        echo "No YAML files found in $DATA_CONFIG_DIR! Skipping..."
        continue
    fi

    for yamlFile in "$DATA_CONFIG_DIR"/*.yaml; do
        echo "Running $SCRIPT with --tune_config $HP_CONFIG and --base_config $yamlFile"

        python "$SCRIPT" --tune_config "$HP_CONFIG" --base_config "$yamlFile" --mode hp
    done
done
