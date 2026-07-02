# Download the latest GPU UTTT checkpoint and run manifest from S3.
#
# Defaults target the UTTT GPU run recorded in artifacts/latest-gpu-run.json.
# Outputs land in artifacts/<runId>/ (round_020.pt + manifest.json).
#
# Manual bootstrap (if this script fails):
#   aws s3 cp s3://nnmcts-artifacts-730335282892-us-west-1/runs/gpu-20260701-192839/manifest.json artifacts/gpu-20260701-192839/manifest.json --region us-west-1
#   aws s3 cp s3://nnmcts-artifacts-730335282892-us-west-1/runs/gpu-20260701-192839/checkpoints/round_020.pt artifacts/gpu-20260701-192839/round_020.pt --region us-west-1
#   python scripts/export_onnx.py
#   python scripts/validate_onnx.py
#   python scripts/generate_uttt_fixtures.py
param(
  [string]$Bucket = "nnmcts-artifacts-730335282892-us-west-1",
  [string]$RunId = "gpu-20260701-192839",
  [string]$CheckpointName = "round_020.pt",
  [string]$RunManifest = "artifacts/latest-gpu-run.json",
  [string]$Profile,
  [string]$Region
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region

if (Test-Path $RunManifest) {
  $runInfo = Get-Content $RunManifest -Raw | ConvertFrom-Json
  if ($runInfo.bucket) { $Bucket = $runInfo.bucket }
  if ($runInfo.runId) { $RunId = $runInfo.runId }
  if ($runInfo.profile) { $Profile = $runInfo.profile }
  if ($runInfo.region) { $Region = $runInfo.region }
}

$outputDir = Join-Path "artifacts" $RunId
$checkpointPath = Join-Path $outputDir $CheckpointName
$manifestPath = Join-Path $outputDir "manifest.json"
$checkpointKey = "runs/$RunId/checkpoints/$CheckpointName"
$manifestKey = "runs/$RunId/manifest.json"

if (-not (Test-Path $outputDir)) {
  New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

function Invoke-AwsDownload {
  param(
    [string]$S3Uri,
    [string]$Destination
  )

  $awsArgs = @(
    "s3", "cp", $S3Uri, $Destination,
    "--profile", $Profile,
    "--region", $Region
  )

  & aws @awsArgs
  return $LASTEXITCODE
}

Write-Host "Downloading GPU UTTT artifacts"
Write-Host "  Bucket:     $Bucket"
Write-Host "  Run:        $RunId"
Write-Host "  Profile:    $Profile"
Write-Host "  Region:     $Region"
Write-Host "  Output dir: $outputDir"
Write-Host ""

$failures = @()

Write-Host "Fetching manifest..."
$manifestExit = Invoke-AwsDownload -S3Uri "s3://$Bucket/$manifestKey" -Destination $manifestPath
if ($manifestExit -ne 0) {
  $failures += "manifest ($manifestKey)"
}
else {
  Write-Host "  Saved $manifestPath"
}

Write-Host "Fetching checkpoint..."
$checkpointExit = Invoke-AwsDownload -S3Uri "s3://$Bucket/$checkpointKey" -Destination $checkpointPath
if ($checkpointExit -ne 0) {
  $failures += "checkpoint ($checkpointKey)"
}
else {
  Write-Host "  Saved $checkpointPath"
}

if ($failures.Count -gt 0) {
  Write-Warning "Download incomplete. Failed: $($failures -join ', ')"
  Write-Host ""
  Write-Host "Manual download:"
  Write-Host "  aws s3 cp s3://$Bucket/$manifestKey $manifestPath --profile $Profile --region $Region"
  Write-Host "  aws s3 cp s3://$Bucket/$checkpointKey $checkpointPath --profile $Profile --region $Region"
  Write-Host ""
  Write-Host "Then export and validate:"
  Write-Host "  python scripts/export_onnx.py --checkpoint $checkpointPath"
  Write-Host "  python scripts/validate_onnx.py --checkpoint $checkpointPath"
  Write-Host "  python scripts/generate_uttt_fixtures.py --checkpoint $checkpointPath"
  exit 1
}

Write-Host ""
Write-Host "Download complete."
exit 0
