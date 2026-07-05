param(
  [string]$Profile,
  [string]$Region,
  [string]$StackName,
  [string]$BaseAmiId,
  [string]$InstanceType = "g4dn.xlarge"
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region -StackName $StackName
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$BaseGpuAmiIds = @{
  "us-west-1" = "ami-0b2f6fd4ed32fc52d"
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
  & aws @CommandArgs --profile $Profile --region $Region 2>$null | Out-Null
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = "Stop"
  return $exitCode -eq 0
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

function Wait-ForInstanceState {
  param(
    [string]$InstanceId,
    [string]$DesiredState,
    [int]$TimeoutSeconds = 1800
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $state = Invoke-AwsCli @(
      "ec2", "describe-instances",
      "--instance-ids", $InstanceId,
      "--query", "Reservations[0].Instances[0].State.Name",
      "--output", "text"
    )
    $state = $state.Trim()
    if ($state -eq $DesiredState) {
      return
    }
    if ($state -in @("terminated", "shutting-down") -and $DesiredState -ne "terminated") {
      throw "Instance $InstanceId entered state '$state' while waiting for '$DesiredState'."
    }
    Write-Host "  instance state: $state (waiting for $DesiredState)"
    Start-Sleep -Seconds 15
  }
  throw "Timed out waiting for instance $InstanceId to reach state '$DesiredState'."
}

function Wait-ForSsmOnline {
  param(
    [string]$InstanceId,
    [int]$TimeoutSeconds = 900
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $ping = Invoke-AwsCli @(
      "ssm", "describe-instance-information",
      "--filters", "Key=InstanceIds,Values=$InstanceId",
      "--query", "InstanceInformationList[0].PingStatus",
      "--output", "text"
    )
    if ($ping.Trim() -eq "Online") {
      return
    }
    Write-Host "  SSM status: $($ping.Trim()) (waiting for Online)"
    Start-Sleep -Seconds 15
  }
  throw "Timed out waiting for SSM agent on instance $InstanceId."
}

function Test-SsmMarker {
  param(
    [string]$InstanceId,
    [string]$MarkerPath
  )

  $commandJson = Invoke-AwsCli @(
    "ssm", "send-command",
    "--instance-ids", $InstanceId,
    "--document-name", "AWS-RunShellScript",
    "--parameters", "commands=[`"test -f $MarkerPath && echo ready`"]",
    "--query", "Command.CommandId",
    "--output", "text"
  )
  $commandId = $commandJson.Trim()

  for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Seconds 10
    $output = Invoke-AwsCli @(
      "ssm", "get-command-invocation",
      "--command-id", $commandId,
      "--instance-id", $InstanceId,
      "--query", "StandardOutputContent",
      "--output", "text"
    )
    if ($output -match "ready") {
      return $true
    }
    $status = Invoke-AwsCli @(
      "ssm", "get-command-invocation",
      "--command-id", $commandId,
      "--instance-id", $InstanceId,
      "--query", "Status",
      "--output", "text"
    )
    if ($status.Trim() -in @("Failed", "Cancelled", "TimedOut")) {
      return $false
    }
  }
  return $false
}

function Wait-ForAmiAvailable {
  param(
    [string]$AmiId,
    [int]$TimeoutSeconds = 3600
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $state = Invoke-AwsCli @(
      "ec2", "describe-images",
      "--image-ids", $AmiId,
      "--query", "Images[0].State",
      "--output", "text"
    )
    $state = $state.Trim()
    if ($state -eq "available") {
      return
    }
    if ($state -eq "failed") {
      throw "AMI $AmiId creation failed."
    }
    Write-Host "  AMI state: $state (waiting for available)"
    Start-Sleep -Seconds 30
  }
  throw "Timed out waiting for AMI $AmiId to become available."
}

if (-not (Invoke-AwsCliAllowFailure @("cloudformation", "describe-stacks", "--stack-name", $StackName))) {
  throw "CloudFormation stack '$StackName' was not found in $Region. Deploy it first: .\scripts\run_cloud_pipeline.ps1 -DeployOnly"
}

$launchTemplate = Get-StackOutput -Key "GpuLaunchTemplateName"
if ([string]::IsNullOrWhiteSpace($launchTemplate) -or $launchTemplate -eq "None") {
  throw "Stack '$StackName' is missing GpuLaunchTemplateName output."
}

if (-not $BaseAmiId) {
  if ($BaseGpuAmiIds.ContainsKey($Region)) {
    $BaseAmiId = $BaseGpuAmiIds[$Region]
  }
  else {
    $BaseAmiId = $BaseGpuAmiIds["us-west-1"]
    Write-Warning "No base GPU AMI mapped for region '$Region'; using us-west-1 fallback $BaseAmiId."
  }
}

$installScriptPath = Join-Path $RepoRoot "cloud\install-gpu-deps.sh"
if (-not (Test-Path -LiteralPath $installScriptPath)) {
  throw "install-gpu-deps.sh not found: $installScriptPath"
}

$installScript = [IO.File]::ReadAllText($installScriptPath).Replace("`r`n", "`n")
$installScriptB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($installScript))

$userData = @'
#!/bin/bash
set -euo pipefail
exec > /var/log/nnmcts-ami-build.log 2>&1
echo "$(date -Is) Starting NNMCTS GPU AMI dependency bake."
echo "__INSTALL_SCRIPT_B64__" | base64 -d > /tmp/install-gpu-deps.sh
chmod +x /tmp/install-gpu-deps.sh
bash /tmp/install-gpu-deps.sh
touch /opt/nnmcts/.ami-build-complete
echo "$(date -Is) AMI bake complete."
'@
$userData = $userData.Replace("__INSTALL_SCRIPT_B64__", $installScriptB64)

$userData = $userData.Replace("`r`n", "`n")
$userDataPath = Join-Path $env:TEMP "nnmcts-ami-user-data.sh"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($userDataPath, $userData, $utf8NoBom)

$imageName = "nnmcts-gpu-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Host "Launching temporary $InstanceType from base AMI $BaseAmiId..."
Write-Host "  Launch template: $launchTemplate"
Write-Host "  Image name:      $imageName"

$instanceJson = Invoke-AwsCli @(
  "ec2", "run-instances",
  "--launch-template", "LaunchTemplateName=$launchTemplate,Version=`$Latest",
  "--image-id", $BaseAmiId,
  "--instance-type", $InstanceType,
  "--user-data", "file://$userDataPath",
  "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=nnmcts-gpu-ami-builder}]",
  "--output", "json"
) | ConvertFrom-Json

$instanceId = $instanceJson.Instances[0].InstanceId
Write-Host "Builder instance: $instanceId"

try {
  Write-Host "Waiting for instance to reach running state..."
  Wait-ForInstanceState -InstanceId $instanceId -DesiredState "running"

  Write-Host "Waiting for SSM agent..."
  Wait-ForSsmOnline -InstanceId $instanceId

  Write-Host "Waiting for dependency install to finish (marker /opt/nnmcts/.ami-build-complete)..."
  $deadline = (Get-Date).AddMinutes(45)
  while ((Get-Date) -lt $deadline) {
    if (Test-SsmMarker -InstanceId $instanceId -MarkerPath "/opt/nnmcts/.ami-build-complete") {
      Write-Host "Dependency install complete."
      break
    }
    Write-Host "  bake still in progress..."
    Start-Sleep -Seconds 30
  }
  if ((Get-Date) -ge $deadline) {
    throw "Timed out waiting for AMI bake to finish on $instanceId. Check /var/log/nnmcts-ami-build.log via SSM."
  }

  Write-Host "Stopping instance before create-image..."
  Invoke-AwsCli @("ec2", "stop-instances", "--instance-ids", $instanceId) | Out-Null
  Wait-ForInstanceState -InstanceId $instanceId -DesiredState "stopped"

  Write-Host "Creating AMI $imageName..."
  $newAmiId = Invoke-AwsCli @(
    "ec2", "create-image",
    "--instance-id", $instanceId,
    "--name", $imageName,
    "--description", "NNMCTS GPU training AMI with pre-baked PyTorch deps",
    "--no-reboot",
    "--query", "ImageId",
    "--output", "text"
  )
  $newAmiId = $newAmiId.Trim()

  Write-Host "Waiting for AMI to become available..."
  Wait-ForAmiAvailable -AmiId $newAmiId

  Write-Host ""
  Write-Host "=== GPU AMI built successfully ==="
  Write-Host "  AMI ID:   $newAmiId"
  Write-Host "  Region:   $Region"
  Write-Host "  Name:     $imageName"
  Write-Host ""
  Write-Host "Next steps:"
  Write-Host "  1. Update config/cloud-training.json gpuAmiIds.$Region to:"
  Write-Host "       `"$newAmiId`""
  Write-Host "  2. Redeploy the CDK stack:"
  Write-Host "       .\scripts\run_cloud_pipeline.ps1 -DeployOnly"
  Write-Host "  3. Launch training as usual (-RunOnly / run_gpu_training.ps1 skips redeploy):"
  Write-Host "       .\scripts\run_gpu_training.ps1"
  Write-Host ""
  Write-Host "Rebuild this AMI when PyTorch, numpy/tqdm, or cloud/install-gpu-deps.sh changes."
}
finally {
  Write-Host "Terminating builder instance $instanceId..."
  Invoke-AwsCliAllowFailure @("ec2", "terminate-instances", "--instance-ids", $instanceId) | Out-Null
}
