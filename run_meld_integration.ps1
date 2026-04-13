Set-Location -LiteralPath $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python).Source
}

& $python ".\meld_dataset_integration.py"
