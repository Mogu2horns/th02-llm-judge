param(
    [string]$PythonPath = 'D:\anaconda\envs\gnnProject\python.exe',
    [string]$CondaPython = 'D:\anaconda\python.exe',
    [switch]$UseCondaCache
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskTemp = Join-Path $PSScriptRoot '.cache\tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
if ($UseCondaCache) {
    $runtimePath = Join-Path $PSScriptRoot '.cache\cuda-runtime'
    $runtimePython = Join-Path $runtimePath 'python.exe'
    $torchPackageRecord = Join-Path $runtimePath 'conda-meta\pytorch-2.5.1-py3.11_cuda12.1_cudnn9_0.json'
    if (-not (Test-Path -LiteralPath $torchPackageRecord)) {
        if (Test-Path -LiteralPath environment-conda-explicit.txt) {
            & $CondaPython -m conda create --offline -y -p $runtimePath --file environment-conda-explicit.txt
        } else {
            & $CondaPython -m conda create --offline -y -p $runtimePath 'python=3.11' 'pytorch=2.5.1' 'pytorch-cuda=12.1'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Cached CUDA runtime creation failed' }
    }
    $PythonPath = $runtimePython
    & $CondaPython -m conda list --explicit -p $runtimePath | Set-Content -LiteralPath environment-conda-explicit.txt -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'Conda package lock failed' }
}
& $PythonPath -c 'import sys; assert sys.version_info[:2] == (3,11)'
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
if ($UseCondaCache) {
    & $PythonPath -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Project CUDA virtual environment creation failed' }
} elseif (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $PythonPath -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
& '.\.venv\Scripts\python.exe' -m ensurepip --upgrade --default-pip
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
$requirementsPath = 'requirements-training.txt'
if (Test-Path -LiteralPath requirements-lock.txt) { $requirementsPath = 'requirements-lock.txt' }
& '.\.venv\Scripts\python.exe' -m pip install --index-url https://pypi.org/simple --timeout 120 --retries 2 -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m th02.environment --lock-path requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'CUDA validation or dependency lock failed' }
