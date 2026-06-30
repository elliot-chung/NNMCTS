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

function Get-RecentInstanceLogs {
  param([string]$Id, [int]$Lines)

  $ssmReady = Invoke-AwsCliAllowFailure @(
    "ssm", "describe-instance-information",
    "--filters", "Key=InstanceIds,Values=$Id",
    "--query", "InstanceInformationList[0].PingStatus",
    "--output", "text"
  )
  if ($ssmReady.ExitCode -ne 0 -or "$($ssmReady.Output)".Trim() -ne "Online") {
    Write-Host "SSM not available for $Id (instance may still be booting or already terminated)."
    return
  }

  $commandJson = Invoke-AwsCli @(
    "ssm", "send-command",
    "--instance-ids", $Id,
    "--document-name", "AWS-RunShellScript",
    "--parameters", "commands=tail -n $Lines /var/log/nnmcts-gpu-train.log",
    "--query", "Command.CommandId",
    "--output", "text"
  )
  $commandId = "$commandJson".Trim()

  $deadline = (Get-Date).AddMinutes(2)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $statusResult = Invoke-AwsCliAllowFailure @(
      "ssm", "get-command-invocation",
      "--command-id", $commandId,
      "--instance-id", $Id,
      "--query", "Status",
      "--output", "text"
    )
    $status = "$($statusResult.Output)".Trim()
    if ($status -in @("Success", "Failed", "Cancelled", "TimedOut")) {
      break
    }
  }

  $invocation = Invoke-AwsCliAllowFailure @(
    "ssm", "get-command-invocation",
    "--command-id", $commandId,
    "--instance-id", $Id,
    "--output", "json"
  )
  if ($invocation.ExitCode -ne 0) {
    Write-Host "Could not fetch logs from instance."
    return
  }

  $payload = $invocation.Output | ConvertFrom-Json
  Write-Host ""
  Write-Host "=== Recent instance log (last $Lines lines) ==="
  if ($payload.StandardOutputContent) {
    Write-Host $payload.StandardOutputContent
  }
  if ($payload.StandardErrorContent) {
    Write-Host $payload.StandardErrorContent
  }
}

function Show-RunStatus {
  param($Metadata)

  $bucket = $Metadata.bucket
  $runId = $Metadata.runId
  $instanceId = $Metadata.instanceId
  $manifestKey = "runs/$runId/manifest.json"

  Write-Host "=== GPU training status ==="
  Write-Host "Run ID:       $runId"
  Write-Host "Instance ID:  $instanceId"
  Write-Host "Launched:     $($Metadata.launchedAt)"
  Write-Host "Manifest:     s3://$bucket/$manifestKey"

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

  if ($state -in @("running", "pending")) {
    Get-RecentInstanceLogs -Id $instanceId -Lines $LogLines
  }
  elseif ($state -in @("terminated", "shutting-down", "stopped", "stopping") -and -not (Test-S3ObjectExists -Bucket $bucket -Key $manifestKey)) {
    Write-Host ""
    Write-Host "Instance ended without uploading a manifest. Logs are only available while the instance is running."
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
