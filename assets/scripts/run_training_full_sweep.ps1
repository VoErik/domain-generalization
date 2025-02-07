param (
    [string]$name
)

$BASE_DIR = "assets\config"
$CONFIG_DIR = "$BASE_DIR\$name"
$SCRIPT = "train.py"

if (-Not (Test-Path -Path $CONFIG_DIR)) {
    Write-Host "Directory $CONFIG_DIR does not exist!"
    exit
}

$yamlFiles = Get-ChildItem -Path $CONFIG_DIR -Recurse -Filter "*.yaml"

if ($yamlFiles.Count -eq 0) {
    Write-Host "No YAML files found in $CONFIG_DIR!"
    exit
}

$yamlFiles | ForEach-Object {
    $subDir = $_.DirectoryName

    Write-Host "Subdirectory: $subDir"
    Write-Host "Running $SCRIPT with --config $($_.FullName)"

    python $SCRIPT --config $_.FullName
}
