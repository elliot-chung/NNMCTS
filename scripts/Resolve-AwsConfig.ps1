function Resolve-AwsConfig {
  param(
    [string]$Profile,
    [string]$Region,
    [string]$StackName
  )

  $repoRoot = Split-Path -Parent $PSScriptRoot
  $configPath = Join-Path $repoRoot "config\local.json"
  $fileConfig = $null
  if (Test-Path $configPath) {
    $fileConfig = Get-Content $configPath -Raw | ConvertFrom-Json
  }

  if (-not $Profile) {
    if ($env:AWS_PROFILE) {
      $Profile = $env:AWS_PROFILE
    }
    elseif ($fileConfig.awsProfile) {
      $Profile = $fileConfig.awsProfile
    }
    else {
      $Profile = "default"
    }
  }

  if (-not $Region) {
    if ($env:AWS_REGION) {
      $Region = $env:AWS_REGION
    }
    elseif ($env:AWS_DEFAULT_REGION) {
      $Region = $env:AWS_DEFAULT_REGION
    }
    elseif ($fileConfig.region) {
      $Region = $fileConfig.region
    }
    else {
      $Region = "us-west-1"
    }
  }

  if (-not $StackName) {
    if ($fileConfig.stackName) {
      $StackName = $fileConfig.stackName
    }
    else {
      $StackName = "NnmctsPipelineStack"
    }
  }

  return @{
    Profile = $Profile
    Region = $Region
    StackName = $StackName
  }
}
