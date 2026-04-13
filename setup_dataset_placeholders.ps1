Set-Location -LiteralPath $PSScriptRoot

$paths = @(
    ".\datasets\CMU-MOSEI",
    ".\datasets\MELD",
    ".\datasets\IEMOCAP",
    ".\datasets\RAVDESS",
    ".\datasets\AffectNet",
    ".\datasets\models\wav2vec2-base",
    ".\datasets\references\facenet",
    ".\datasets\references\facs"
)

foreach ($path in $paths) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

Write-Host "Dataset placeholder folders are ready."
