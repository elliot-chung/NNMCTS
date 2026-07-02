param(
  [string]$Profile,
  [string]$Region,
  [string]$StackName,
  [string]$RunId,
  [string]$InstanceId,
  [string]$MetadataPath,
  [int]$LogLines = 50,
  [switch]$Follow
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region -StackName $StackName
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $MetadataPath) {
  $MetadataPath = Join-Path $RepoRoot "artifacts\latest-gpu-run.json"
}

function Invoke-AwsCli {
  param([string[]]$CommandArgs)
  & aws @CommandArgs --profile $Profile --region $Region
  if ($LASTEXITCODE -ne 0) {
    throw "aws command failed: aws $($CommandArgs -join ' ')"
  }
}

function Invoke-AwsCliAllowFailure {
  param([string[]]$CommandArgs)
  $ErrorActionPreference = "SilentlyContinue"
  $output = & aws @CommandArgs --profile $Profile --region $Region 2>&1
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = "Stop"
  return @{ ExitCode = $exitCode; Output = $output }
}

function Test-S3ObjectExists {
  param([string]$Bucket, [string]$Key)
  $result = Invoke-AwsCliAllowFailure @(
    "s3api", "head-object",
    "--bucket", $Bucket,
    "--key", $Key
  )
  return $result.ExitCode -eq 0
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

function Read-RunMetadata {
  if (-not (Test-Path $MetadataPath)) {
    throw "Run metadata not found at $MetadataPath. Launch a GPU run with .\scripts\run_gpu_training.ps1 first."
  }
  return Get-Content $MetadataPath -Raw | ConvertFrom-Json
}

function Get-InstanceState {
  param([string]$Id)
  $result = Invoke-AwsCliAllowFailure @(
    "ec2", "describe-instances",
    "--instance-ids", $Id,
    "--query", "Reservations[0].Instances[0].State.Name",
    "--output", "text"
  )
  if ($result.ExitCode -ne 0) {
    return "unknown"
  }
  return "$($result.Output)".Trim()
}

function Get-CloudWatchLogs {
  param(
    [string]$LogGroup,
    [string]$StreamName,
    [int]$Lines
  )

  $result = Invoke-AwsCliAllowFailure @(
    "logs", "get-log-events",
    "--log-group-name", $LogGroup,
    "--log-stream-name", $StreamName,
    "--limit", "$Lines",
    "--start-from-head", "false",
    "--output", "json"
  )
  if ($result.ExitCode -ne 0) {
    $errorText = "$($result.Output)"
    if ($errorText -match "ResourceNotFoundException") {
      Write-Host ""
      Write-Host "=== Recent CloudWatch logs ==="
      Write-Host "No log events yet (instance may still be booting)."
      return
    }
    Write-Host ""
    Write-Host "=== Recent CloudWatch logs ==="
    Write-Host "Could not fetch logs from CloudWatch: $errorText"
    return
  }

  $payload = $result.Output | ConvertFrom-Json
  $events = @($payload.events)
  if ($events.Count -eq 0) {
    Write-Host ""
    Write-Host "=== Recent CloudWatch logs ==="
    Write-Host "No log events yet (instance may still be booting)."
    return
  }

  Write-Host ""
  Write-Host "=== Recent CloudWatch logs (last $($events.Count) events) ==="
  foreach ($event in ($events | Sort-Object timestamp)) {
    Write-Host $event.message
  }
}

function Resolve-LogGroupName {
  param($Metadata)

  if ($Metadata.logGroupName) {
    return "$($Metadata.logGroupName)"
  }

  $result = Invoke-AwsCliAllowFailure @(
    "cloudformation", "describe-stacks",
    "--stack-name", $StackName,
    "--query", "Stacks[0].Outputs[?OutputKey=='GpuTrainingLogGroupName'].OutputValue",
    "--output", "text"
  )
  if ($result.ExitCode -eq 0 -and "$($result.Output)".Trim()) {
    return "$($result.Output)".Trim()
  }

  return "/nnmcts/gpu-training"
}

function Show-RunStatus {
  param($Metadata)

  $bucket = $Metadata.bucket
  $runId = $Metadata.runId
  $instanceId = $Metadata.instanceId
  $manifestKey = "runs/$runId/manifest.json"
  $logGroup = Resolve-LogGroupName -Metadata $Metadata

  Write-Host "=== GPU training status ==="
  Write-Host "Run ID:       $runId"
  Write-Host "Instance ID:  $instanceId"
  Write-Host "Launched:     $($Metadata.launchedAt)"
  Write-Host "Manifest:     s3://$bucket/$manifestKey"
  Write-Host "CloudWatch:   $logGroup / $instanceId"

  $state = Get-InstanceState -Id $instanceId
  Write-Host "Instance:     $state"

  if (Test-S3ObjectExists -Bucket $bucket -Key $manifestKey) {
    Write-Host ""
    Write-Host "=== Manifest ==="
    Invoke-AwsCli @("s3", "cp", "s3://$bucket/$manifestKey", "-") | Write-Host

    $checkpointPrefix = "runs/$runId/checkpoints/"
    Write-Host ""
    Write-Host "=== Checkpoints ==="
    Invoke-AwsCli @("s3", "ls", "s3://$bucket/$checkpointPrefix")
  }
  else {
    Write-Host "Manifest:     not uploaded yet"
  }

  Get-CloudWatchLogs -LogGroup $logGroup -StreamName $instanceId -Lines $LogLines

  if ($state -in @("terminated", "shutting-down", "stopped", "stopping") -and -not (Test-S3ObjectExists -Bucket $bucket -Key $manifestKey)) {
    Write-Host ""
    Write-Host "Instance ended without uploading a manifest. Check CloudWatch logs above or s3://$bucket/runs/$runId/gpu-train.log after upload."
  }
}

$metadata = Read-RunMetadata
if ($RunId) { $metadata.runId = $RunId }
if ($InstanceId) { $metadata.instanceId = $InstanceId }

if ($Follow) {
  while ($true) {
    Clear-Host
    Show-RunStatus -Metadata $metadata
    $state = Get-InstanceState -Id $metadata.instanceId
    $manifestKey = "runs/$($metadata.runId)/manifest.json"
    if (Test-S3ObjectExists -Bucket $metadata.bucket -Key $manifestKey) {
      Write-Host ""
      Write-Host "Training finished."
      break
    }
    if ($state -in @("terminated", "shutting-down", "stopped", "stopping")) {
      Write-Host ""
      Write-Host "Instance ended."
      break
    }
    Start-Sleep -Seconds 30
  }
}
else {
  Show-RunStatus -Metadata $metadata
}
