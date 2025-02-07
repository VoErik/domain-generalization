#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_directory_name>"
    exit 1
fi

BASE_DIR="assets/config"
CONFIG_DIR="$BASE_DIR/$1"
SCRIPT="train.py"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Directory $CONFIG_DIR does not exist!"
    exit 1
fi

mapfile -t yamlFiles < <(find "$CONFIG_DIR" -type f -name "*.yaml")

if [ ${#yamlFiles[@]} -eq 0 ]; then
    echo "No YAML files found in $CONFIG_DIR!"
    exit 1
fi

for config in "${yamlFiles[@]}"; do
    subDir=$(dirname "$config")  # Get the subdirectory name
    echo "Subdirectory: $subDir"
    echo "Running $SCRIPT with --config $config"
    python "$SCRIPT" --config "$config"
done
