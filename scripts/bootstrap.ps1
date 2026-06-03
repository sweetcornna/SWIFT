param(
    [ValidateSet("quick", "train", "pybullet")]
    [string]$Mode = "quick",

    [string]$VenvPath = ".venv",

    [string]$PyBulletRoot = "",

    [string]$PyBulletVenv = ".venv-pybullet",

    [string]$Python = "python",

    [switch]$DryRun,

    [switch]$NoVerify
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$bootstrap = Join-Path $PSScriptRoot "bootstrap.py"

$arguments = @(
    $bootstrap,
    "--mode", $Mode,
    "--venv-path", $VenvPath,
    "--python", $Python
)

if ($Mode -eq "pybullet") {
    if ($PyBulletRoot -ne "") {
        $arguments += @("--pybullet-root", $PyBulletRoot)
    }
    $arguments += @("--pybullet-venv", $PyBulletVenv)
}

if ($DryRun) {
    $arguments += "--dry-run"
}

if ($NoVerify) {
    $arguments += "--no-verify"
}

Push-Location $repoRoot
try {
    & $Python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
