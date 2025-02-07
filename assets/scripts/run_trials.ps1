param (
    [string]$base_config,
    [string]$tune_config
)

$ProjectRoot = Get-Location
$HP_CONFIG = Join-Path $ProjectRoot "assets\config\hp-tuning\hp_search_spaces\$tune_config"
$DATA_CONFIG_DIR = Join-Path $ProjectRoot "assets\config\hp-tuning\base-models\$base_config"
$SCRIPT = Join-Path $ProjectRoot "tune_config.py"

if (-Not (Test-Path -Path $DATA_CONFIG_DIR)) {
    Write-Host "Directory $DATA_CONFIG_DIR does not exist!"
    exit 1
}

$yamlFiles = Get-ChildItem -Path $DATA_CONFIG_DIR -Filter "*.yaml"
if ($yamlFiles.Count -eq 0) {
    Write-Host "No YAML files found in $DATA_CONFIG_DIR!"
    exit 1
}

$yamlFiles | ForEach-Object {
    $model_config = $_.FullName
    Write-Host "Running $SCRIPT with --tune_config $HP_CONFIG and --base_config $model_config"

    python $SCRIPT --tune_config $HP_CONFIG --base_config $model_config --mode hp
}




