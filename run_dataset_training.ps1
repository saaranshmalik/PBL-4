Set-Location -LiteralPath $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python).Source
}

$argList = [System.Collections.Generic.List[string]]::new()
if ($args -notcontains "--dataset") {
    $argList.Add("--dataset")
    $argList.Add("meld")
}

foreach ($arg in $args) {
    $argList.Add($arg)
}

& $python ".\dataset_training_module.py" @argList
