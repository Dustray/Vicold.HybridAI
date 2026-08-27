# Run the Qwen inference demo on the ROCm-capable GPU (device 1).
# This script activates the Python venv, sets HIP_VISIBLE_DEVICES and PATH,
# then invokes the compiled demo executable.
param(
    [string]$ModelDir = "E:/models/Qwen3.8-27B-FP8",
    [string]$Backend = "gpu",
    [int]$MaxNewTokens = 1
)

$ErrorActionPreference = "Stop"

$venvPath = "C:\Users\yinxi\.venv"
$rocmBin = "C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin"

& "$venvPath\Scripts\Activate.ps1"

$env:HIP_VISIBLE_DEVICES = "1"
$env:PATH = "$rocmBin;" + $env:PATH

$exe = "d:\Vicold\Vicold.HybridAI\build-debug\bin\Debug\qwen_infer.exe"

Write-Host "Running: $exe $ModelDir $Backend $MaxNewTokens" -ForegroundColor Cyan
& $exe $ModelDir $Backend $MaxNewTokens
