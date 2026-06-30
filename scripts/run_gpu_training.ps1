param(
  [string]$Profile,
  [string]$Region,
  [string]$StackName
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region -StackName $StackName
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-AwsCli {
  param([string[]]$CommandArgs)
  & aws @CommandArgs --profile $Profile --region $Region
  if ($LASTEXITCODE -ne 0) {
    throw "aws command failed: aws $($CommandArgs -join ' ')"
  }
}

function Get-StackOutput {
  param([string]$Key)
  $value = Invoke-AwsCli @(
    "cloudformation", "describe-stacks",
    "--stack-name", $StackName,
    "--query", "Stacks[0].Outputs[?OutputKey=='$Key'].OutputValue",
    "--output", "text"
  )
  return $value.Trim()
}

$bucket = Get-StackOutput -Key "ArtifactsBucketName"
$launchTemplate = Get-StackOutput -Key "GpuLaunchTemplateName"
$runId = "gpu-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$sourceKey = "source/nnmcts-$runId.zip"
$launchedAt = (Get-Date).ToString("o")

Write-Host "Packaging source..."
$zipPath = & (Join-Path $PSScriptRoot "package_source.ps1")

Write-Host "Uploading source to s3://$bucket/$sourceKey"
Invoke-AwsCli @("s3", "cp", $zipPath, "s3://$bucket/$sourceKey")

Write-Host "Launching GPU training instance (g4dn.xlarge, 1h training / 90m instance cap)..."
$instanceJson = Invoke-AwsCli @(
  "ec2", "run-instances",
  "--launch-template", "LaunchTemplateName=$launchTemplate",
  "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=nnmcts-gpu-training},{Key=nnmcts-bucket,Value=$bucket},{Key=nnmcts-source-key,Value=$sourceKey},{Key=nnmcts-run-id,Value=$runId}]",
  "--output", "json"
) | ConvertFrom-Json

$instanceId = $instanceJson.Instances[0].InstanceId

$artifactsDir = Join-Path $RepoRoot "artifacts"
if (-not (Test-Path $artifactsDir)) {
  New-Item -ItemType Directory -Path $artifactsDir | Out-Null
}

$metadataPath = Join-Path $artifactsDir "latest-gpu-run.json"
$metadata = [ordered]@{
  runId = $runId
  instanceId = $instanceId
  bucket = $bucket
  sourceKey = $sourceKey
  manifestKey = "runs/$runId/manifest.json"
  profile = $Profile
  region = $Region
  stackName = $StackName
  launchedAt = $launchedAt
}
$metadata | ConvertTo-Json | Set-Content -Path $metadataPath -Encoding utf8

Write-Host ""
Write-Host "GPU training launched."
Write-Host "  Instance ID:  $instanceId"
Write-Host "  Run ID:       $runId"
Write-Host "  Manifest:     s3://$bucket/runs/$runId/manifest.json"
Write-Host "  Run metadata: $metadataPath"
Write-Host ""
Write-Host "Check status and recent logs:"
Write-Host "  .\scripts\check_gpu_training.ps1"
Write-Host ""
Write-Host "Poll until complete:"
Write-Host "  .\scripts\check_gpu_training.ps1 -Follow"
Write-Host ""
Write-Host "Redeploy the CDK stack if you changed cloud/gpu-train.sh since the last deploy."
