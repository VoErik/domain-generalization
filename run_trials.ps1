# Define variables
$HP_CONFIG = "config\hp_search_space.yaml"
$DATA_CONFIG_DIR = "config\base\$args"
$SCRIPT = "tune_config.py"

# Check if the directory exists
if (-Not (Test-Path -Path $DATA_CONFIG_DIR)) {
    Write-Host "Directory $DATA_CONFIG_DIR does not exist!"
    exit
}

# Check if the directory contains YAML files
$yamlFiles = Get-ChildItem -Path $DATA_CONFIG_DIR -Filter "*.yaml"
if ($yamlFiles.Count -eq 0) {
    Write-Host "No YAML files found in $DATA_CONFIG_DIR!"
    exit
}

# Loop through the YAML files and execute the command
$yamlFiles | ForEach-Object {
    $data_config = $_.FullName
    Write-Host "Running $SCRIPT with --hp_config $HP_CONFIG and --data_config $data_config"

    # Run the Python script with arguments
    python $SCRIPT --hp_config $HP_CONFIG --data_config $data_config --mode hp
}


