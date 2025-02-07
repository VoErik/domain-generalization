param (
    [string]$base_config,
    [string]$tune_config
)

$ProjectRoot = Get-Location
$HP_CONFIG = Join-Path $ProjectRoot "assets\config\hp-tuning\hp_search_spaces\$tune_config"
$DATA_CONFIG_DIR = Join-Path $ProjectRoot "assets\config\hp-tuning\base-models\$base_config"
$SCRIPT = Join-Path $ProjectRoot "tune_config.py"

if (-Not (Test-Path -Path $BASE_MODELS_DIR)) {
    Write-Host "Base models directory $BASE_MODELS_DIR does not exist!"
    exit 1
}

$subDirs = Get-ChildItem -Path $BASE_MODELS_DIR -Directory

if ($subDirs.Count -eq 0) {
    Write-Host "No subdirectories found in $BASE_MODELS_DIR!"
    exit 1
}

foreach ($subDir in $subDirs) {
    $DATA_CONFIG_DIR = $subDir.FullName

    $yamlFiles = Get-ChildItem -Path $DATA_CONFIG_DIR -Filter "*.yaml"

    if ($yamlFiles.Count -eq 0) {
        Write-Host "No YAML files found in $DATA_CONFIG_DIR! Skipping..."
        continue
    }

    foreach ($yamlFile in $yamlFiles) {
        $model_config = $yamlFile.FullName
        Write-Host "Running $SCRIPT with --tune_config $HP_CONFIG and --base_config $model_config"

        python $SCRIPT --tune_config $HP_CONFIG --base_config $model_config --mode hp
    }
}