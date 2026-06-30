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

function Invoke-AwsCli {
  param([string[]]$CommandArgs)
  & aws @CommandArgs --profile $Profile --region $Region
  if ($LASTEXITCODE -ne 0) {
    throw "aws command failed: aws $($CommandArgs -join ' ')"
  }
}

function Test-S3ObjectExists {
  param([string]$Bucket, [string]$Key)
  $ErrorActionPreference = "SilentlyContinue"
  & aws s3api head-object --bucket $Bucket --key $Key --profile $Profile --region $Region 2>$null | Out-Null
  $exists = $LASTEXITCODE -eq 0
  $ErrorActionPreference = "Stop"
  return $exists
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

$RepoRoot = Split-Path -Parent $PSScriptRoot
$bucket = Get-StackOutput -Key "ArtifactsBucketName"
$launchTemplate = Get-StackOutput -Key "GpuLaunchTemplateName"
$runId = "gpu-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$sourceKey = "source/nnmcts-$runId.zip"

Write-Host "Packaging source..."
$zipPath = & (Join-Path $PSScriptRoot "package_source.ps1")

Write-Host "Uploading source to s3://$bucket/$sourceKey"
Invoke-AwsCli @("s3", "cp", $zipPath, "s3://$bucket/$sourceKey")

Write-Host "Launching GPU training instance (g4dn.xlarge, max 1 hour)..."
$instanceJson = Invoke-AwsCli @(
  "ec2", "run-instances",
  "--launch-template", "LaunchTemplateName=$launchTemplate",
  "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=nnmcts-gpu-training},{Key=nnmcts-bucket,Value=$bucket},{Key=nnmcts-source-key,Value=$sourceKey},{Key=nnmcts-run-id,Value=$runId}]",
  "--output", "json"
) | ConvertFrom-Json

$instanceId = $instanceJson.Instances[0].InstanceId
Write-Host "Instance ID: $instanceId"
Write-Host "Run ID: $runId"
Write-Host "Training log on instance: /var/log/nnmcts-gpu-train.log"
Write-Host "Waiting for manifest at s3://$bucket/runs/$runId/manifest.json ..."

$deadline = (Get-Date).AddHours(1.25)
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 30
  $state = (Invoke-AwsCli @(
    "ec2", "describe-instances",
    "--instance-ids", $instanceId,
    "--query", "Reservations[0].Instances[0].State.Name",
    "--output", "text"
  )).Trim()
  Write-Host "  instance=$state"

  if (Test-S3ObjectExists -Bucket $bucket -Key "runs/$runId/manifest.json") {
    Write-Host "Training complete."
    Invoke-AwsCli @("s3", "cp", "s3://$bucket/runs/$runId/manifest.json", "-")
    Invoke-AwsCli @("s3", "ls", "s3://$bucket/runs/$runId/checkpoints/")
    exit 0
  }

  if ($state -in @("terminated", "shutting-down", "stopped", "stopping")) {
    throw "Instance ended before manifest appeared. Check /var/log/nnmcts-gpu-train.log via SSM or CloudWatch."
  }
}

throw "Timed out waiting for training manifest."
