param(
    [string]$Config = "uwb_tcn_residual_seq10_ep10_test",
    [string]$Dataset = "otb100_uwb",
    [string]$Sequence = "BlurBody",
    [int]$DebugLevel = 1,
    [int]$NumGpus = 1,
    [int]$VisdomPort = 8097,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Resolve repo root.
$ScriptPath = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
    $RepoRoot = Resolve-Path "."
}
else {
    $RepoRoot = Resolve-Path (Join-Path $ScriptPath "..")
}
Set-Location $RepoRoot

# Activate conda env.
if (-not (Test-Path -LiteralPath 'D:\DeepLearning\Miniconda\shell\condabin\conda-hook.ps1')) {
    throw 'Conda hook not found: D:\DeepLearning\Miniconda\shell\condabin\conda-hook.ps1'
}

& 'D:\DeepLearning\Miniconda\shell\condabin\conda-hook.ps1'
conda activate ostrack

# Check visdom server.
$VisdomUrl = "http://localhost:$VisdomPort"
try {
    Invoke-WebRequest -Uri $VisdomUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
    Write-Host "Visdom is available: $VisdomUrl"
}
catch {
    Write-Warning "Visdom is not reachable at $VisdomUrl. The tracker may fall back to local debug images."
}

$ArgsList = @(
    "tracking\test.py",
    "ugtrack",
    $Config,
    "--dataset_name", $Dataset,
    "--sequence", $Sequence,
    "--debug", "$DebugLevel",
    "--num_gpus", "$NumGpus"
)

Write-Host "Command:"
Write-Host "python $($ArgsList -join ' ')"

if ($DryRun) {
    Write-Host "DryRun enabled. Command was not executed."
    exit 0
}

python @ArgsList
