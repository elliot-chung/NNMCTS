param(
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$OutputPath = "$env:TEMP\nnmcts-source.zip",
  [string]$CheckpointPath
)

$ErrorActionPreference = "Stop"
$pythonArgs = @(
  (Join-Path $PSScriptRoot "package_source.py"),
  "--repo-root", $RepoRoot,
  "--output", $OutputPath
)
if ($CheckpointPath) {
  $resolvedCheckpoint = Resolve-Path -LiteralPath $CheckpointPath
  $pythonArgs += @("--checkpoint", $resolvedCheckpoint.Path)
}
python @pythonArgs
