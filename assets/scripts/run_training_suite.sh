#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_directory_name>"
    exit 1
fi

CONFIG_DIR="assets/config/$1"
SCRIPT="train.py"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Directory $CONFIG_DIR does not exist!"
    exit 1
fi

yamlFiles=("$CONFIG_DIR"/*.yaml)

if [ ${#yamlFiles[@]} -eq 0 ]; then
    echo "No YAML files found in $CONFIG_DIR!"
    exit 1
fi

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running $SCRIPT with --config $config"
    python "$SCRIPT" --config "$config"
done
