#Requires -Version 5.1
<#
.SYNOPSIS
    Download dependency source archives for offline/local CMake build.
.DESCRIPTION
    Downloads fmt, spdlog, nlohmann_json and googletest release archives
    into third_party/deps_local. Set HTTP_PROXY/HTTPS_PROXY if needed.
#>
$ErrorActionPreference = 'Stop'

# Proxy settings for GitHub access
$env:HTTP_PROXY  = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'

$deps = @(
    @{ Name = 'fmt';           Version = '11.0.2';  Url = 'https://github.com/fmtlib/fmt/archive/refs/tags/11.0.2.zip';          Dir = 'fmt-11.0.2' },
    @{ Name = 'spdlog';        Version = '1.14.1';  Url = 'https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.zip';     Dir = 'spdlog-1.14.1' },
    @{ Name = 'nlohmann_json'; Version = '3.11.3';  Url = 'https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip';      Dir = 'json-3.11.3' },
    @{ Name = 'googletest';    Version = '1.15.2';  Url = 'https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip';    Dir = 'googletest-1.15.2' }
)

$root = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path (Join-Path $root 'third_party') 'deps_local'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

foreach ($dep in $deps) {
    $file = Join-Path $outDir ($dep.Name + '-' + $dep.Version + '.zip')
    if (Test-Path $file) {
        Write-Host "[$($dep.Name)] archive already exists, skipping download."
        continue
    }
    Write-Host "[$($dep.Name)] downloading $($dep.Url) ..."
    try {
        Invoke-WebRequest -Uri $dep.Url -OutFile $file -UseBasicParsing -MaximumRedirection 5 -TimeoutSec 120
        Write-Host "[$($dep.Name)] saved to $file"
    }
    catch {
        Write-Error "[$($dep.Name)] failed to download: $_"
    }
}

Write-Host "`nAll dependency archives are located in: $outDir"
