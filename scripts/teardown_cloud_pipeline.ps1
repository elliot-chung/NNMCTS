param(
  [string]$Profile,
  [string]$Region,
  [string]$StackName,
  [string]$GpuInstanceTag = "nnmcts-gpu-training",
  [switch]$KeepArtifacts,
  [switch]$Force
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region -StackName $StackName
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$InfraDir = Join-Path $RepoRoot "infra"

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

function Test-StackExists {
  return Invoke-AwsCliAllowFailure @(
    "cloudformation", "describe-stacks",
    "--stack-name", $StackName
  )
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

function Get-ArtifactsBucketName {
  if (Test-StackExists) {
    return Get-StackOutput -Key "ArtifactsBucketName"
  }

  $account = (Invoke-AwsCli @(
    "sts", "get-caller-identity",
    "--query", "Account",
    "--output", "text"
  )).Trim()
  return "nnmcts-artifacts-$account-$Region"
}

function Stop-GpuInstances {
  $instancesJson = Invoke-AwsCli @(
    "ec2", "describe-instances",
    "--filters",
    "Name=tag:Name,Values=$GpuInstanceTag",
    "Name=instance-state-name,Values=pending,running,stopping,stopped",
    "--query", "Reservations[].Instances[].InstanceId",
    "--output", "json"
  ) | ConvertFrom-Json

  $instanceIds = @($instancesJson) | Where-Object { $_ }
  if ($instanceIds.Count -eq 0) {
    Write-Host "No active GPU training instances found."
    return @()
  }

  Write-Host "Terminating GPU instances: $($instanceIds -join ', ')"
  Invoke-AwsCli (@(
    "ec2", "terminate-instances",
    "--instance-ids"
  ) + @($instanceIds)) | Out-Null
  return $instanceIds
}

function Wait-InstancesTerminated {
  param([string[]]$InstanceIds)

  if ($InstanceIds.Count -eq 0) {
    return
  }

  Write-Host "Waiting for instances to terminate..."
  $deadline = (Get-Date).AddMinutes(10)
  while ((Get-Date) -lt $deadline) {
    $states = Invoke-AwsCli (@(
      "ec2", "describe-instances",
      "--instance-ids"
    ) + @($InstanceIds) + @(
      "--query", "Reservations[].Instances[].State.Name",
      "--output", "json"
    )) | ConvertFrom-Json

    $active = $states | Where-Object { $_ -notin @("terminated", "shutting-down") }
    if (-not $active -or $active.Count -eq 0) {
      Write-Host "All instances terminated."
      return
    }

    Write-Host "  still terminating: $($active -join ', ')"
    Start-Sleep -Seconds 15
  }

  Write-Warning "Timed out waiting for instance termination. Continuing teardown."
}


function Remove-VersionedS3Bucket {
  param([string]$Bucket)

  if (-not (Invoke-AwsCliAllowFailure @("s3api", "head-bucket", "--bucket", $Bucket))) {
    Write-Host "Artifacts bucket '$Bucket' does not exist."
    return
  }

  Write-Host "Emptying artifacts bucket s3://$Bucket ..."
  $keyMarker = $null
  $versionIdMarker = $null

  while ($true) {
    $listArgs = @("s3api", "list-object-versions", "--bucket", $Bucket, "--output", "json")
    if ($keyMarker) {
      $listArgs += @("--key-marker", $keyMarker, "--version-id-marker", $versionIdMarker)
    }

    $page = Invoke-AwsCli $listArgs | ConvertFrom-Json
    $objects = @()

    if ($page.Versions) {
      foreach ($version in @($page.Versions)) {
        $objects += @{ Key = $version.Key; VersionId = $version.VersionId }
      }
    }
    if ($page.DeleteMarkers) {
      foreach ($marker in @($page.DeleteMarkers)) {
        $objects += @{ Key = $marker.Key; VersionId = $marker.VersionId }
      }
    }

    if ($objects.Count -gt 0) {
      $payload = @{ Objects = $objects; Quiet = $true } | ConvertTo-Json -Compress -Depth 4
      $tempFile = [System.IO.Path]::GetTempFileName()
      try {
        $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($tempFile, $payload, $utf8NoBom)
        $deleteFile = "file://" + ($tempFile -replace '\\', '/')
        Invoke-AwsCli @("s3api", "delete-objects", "--bucket", $Bucket, "--delete", $deleteFile) | Out-Null
      }
      finally {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
      }
    }

    if ($page.IsTruncated) {
      $keyMarker = $page.NextKeyMarker
      $versionIdMarker = $page.NextVersionIdMarker
      continue
    }
    break
  }

  Write-Host "Deleting bucket s3://$Bucket ..."
  Invoke-AwsCli @("s3api", "delete-bucket", "--bucket", $Bucket)
}

function Destroy-Stack {
  if (-not (Test-StackExists)) {
    Write-Host "CloudFormation stack '$StackName' not found."
    return
  }

  if (-not (Test-Path (Join-Path $InfraDir "node_modules"))) {
    Write-Host "Installing CDK dependencies..."
    Push-Location $InfraDir
    try {
      npm install --silent
    }
    finally {
      Pop-Location
    }
  }

  $account = (Invoke-AwsCli @("sts", "get-caller-identity", "--query", "Account", "--output", "text")).Trim()
  $env:CDK_DEFAULT_ACCOUNT = $account
  $env:CDK_DEFAULT_REGION = $Region

  Write-Host "Destroying CloudFormation stack '$StackName'..."
  Push-Location $InfraDir
  try {
    npx cdk destroy $StackName --profile $Profile --force
    if ($LASTEXITCODE -ne 0) {
      throw "cdk destroy failed with exit code $LASTEXITCODE"
    }
  }
  finally {
    Pop-Location
  }
}

if (-not $Force) {
  $target = "NNMCTS cloud pipeline in $Region (profile: $Profile)"
  if (-not $KeepArtifacts) {
    $target += " including the S3 artifacts bucket"
  }
  $answer = Read-Host "This will terminate GPU instances and remove $target. Continue? [y/N]"
  if ($answer -notin @("y", "Y", "yes", "Yes")) {
    Write-Host "Teardown cancelled."
    exit 0
  }
}

Write-Host "=== NNMCTS cloud pipeline teardown ==="
$terminatedInstances = Stop-GpuInstances
Wait-InstancesTerminated -InstanceIds $terminatedInstances

$artifactsBucket = Get-ArtifactsBucketName
Destroy-Stack

if (-not $KeepArtifacts) {
  Remove-VersionedS3Bucket -Bucket $artifactsBucket
}
else {
  Write-Host "Keeping artifacts bucket: s3://$artifactsBucket"
}

Write-Host ""
Write-Host "Teardown complete."
Write-Host "Note: the CDK bootstrap stack (CDKToolkit) was not removed."
