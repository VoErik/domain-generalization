param (
    [string]$name
)

$CONFIG_DIR = "assets\config\$name"
$SCRIPT = "train.py"

if (-Not (Test-Path -Path $CONFIG_DIR)) {
    Write-Host "Directory $CONFIG_DIR does not exist!"
    exit
}

$yamlFiles = Get-ChildItem -Path $CONFIG_DIR -Filter "*.yaml"
if ($yamlFiles.Count -eq 0) {
    Write-Host "No YAML files found in $CONFIG_DIR!"
    exit
}

$yamlFiles | ForEach-Object {
    $config = $_.FullName
    Write-Host "Running $SCRIPT with --config $config"
    python $SCRIPT --config $config
}