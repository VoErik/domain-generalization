$HP_CONFIG = "config\hp_search_space.yaml"
$SCRIPT = "tune_config.py"
$CONFIG_DIR = "configs\base"

Get-ChildItem -Path $CONFIG_DIR -Recurse -Filter *.yaml | ForEach-Object {
    Write-Host "Running $SCRIPT with --hp_config $HP_CONFIG and --data_config $($_.FullName)"
    python $SCRIPT --hp_config $HP_CONFIG --data_config $($_.FullName) --mode hp
}
