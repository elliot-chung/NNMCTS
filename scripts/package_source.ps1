param(
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$OutputPath = "$env:TEMP\nnmcts-source.zip"
)

$ErrorActionPreference = "Stop"
python (Join-Path $PSScriptRoot "package_source.py") --repo-root $RepoRoot --output $OutputPath
