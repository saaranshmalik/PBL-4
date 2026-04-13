Set-Location -LiteralPath $PSScriptRoot
$python = (Get-Command python).Source
& $python -u ".\dataset_testing_backend.py"
