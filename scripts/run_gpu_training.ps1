param(
  [string]$Profile,
  [string]$Region,
  [string]$StackName,
  [string]$ConfigPath,
  [ValidateSet("gpuSmoke", "gpu")]
  [string]$TrainingProfile = "gpu",
  [switch]$SmokeThenTrain,
  [string]$GameType,
  [int]$Rounds = 0,
  [int]$GamesPerRound = 0,
  [int]$Epochs = 0,
  [int]$BatchSize = 0,
  [int]$MctsIters = 0,
  [string]$Player1Type,
  [string]$Player2Type,
  [int]$SelfPlayWorkers = 0,
  [int]$MaxTrainingSeconds = 0,
  [int]$MaxInstanceSeconds = 0,
  [switch]$Wait
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region -StackName $StackName
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

. (Join-Path $PSScriptRoot "Resolve-CloudTrainingConfig.ps1")
$trainingConfig = Resolve-CloudTrainingConfig -Profile $TrainingProfile -ConfigPath $ConfigPath -GameType $GameType -Rounds $Rounds -GamesPerRound $GamesPerRound -Epochs $Epochs -BatchSize $BatchSize -MctsIters $MctsIters -Player1Type $Player1Type -Player2Type $Player2Type -SelfPlayWorkers $SelfPlayWorkers -MaxTrainingSeconds $MaxTrainingSeconds -MaxInstanceSeconds $MaxInstanceSeconds

$smokeConfig = $null
if ($SmokeThenTrain) {
  $smokeConfig = Resolve-CloudTrainingConfig -Profile gpuSmoke -ConfigPath $ConfigPath -GameType $GameType
  $TrainingProfile = "gpu"
}

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

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

function Format-Ec2TagSpecifications {
  param(
    [array]$Tags
  )

  $formattedTags = @()
  foreach ($tag in $Tags) {
    $formattedTags += "{Key=$($tag.Key),Value=$($tag.Value)}"
  }
  return "ResourceType=instance,Tags=[$($formattedTags -join ',')]"
}

function Test-S3ObjectExists {
  param([string]$Bucket, [string]$Key)
  return Invoke-AwsCliAllowFailure @(
    "s3api", "head-object",
    "--bucket", $Bucket,
    "--key", $Key
  )
}

function Get-InstanceState {
  param([string]$Id)
  if (-not (Invoke-AwsCliAllowFailure @("ec2", "describe-instances", "--instance-ids", $Id))) {
    return "unknown"
  }
  $state = Invoke-AwsCli @(
    "ec2", "describe-instances",
    "--instance-ids", $Id,
    "--query", "Reservations[0].Instances[0].State.Name",
    "--output", "text"
  )
  return $state.Trim()
}

function Wait-ForGpuRun {
  param(
    [string]$Bucket,
    [string]$RunId,
    [string]$InstanceId,
    [string]$Label
  )

  $manifestKey = "runs/$RunId/manifest.json"
  Write-Host "Waiting for $Label to finish (manifest: s3://$Bucket/$manifestKey)..."

  while ($true) {
    if (Test-S3ObjectExists -Bucket $Bucket -Key $manifestKey) {
      $manifestJson = Invoke-AwsCli @("s3", "cp", "s3://$Bucket/$manifestKey", "-")
      $manifest = $manifestJson | ConvertFrom-Json
      Write-Host ""
      Write-Host "=== $Label manifest ==="
      Write-Host $manifestJson
      if ($manifest.status -ne "complete") {
        throw "$Label finished with status '$($manifest.status)'. Check s3://$Bucket/runs/$RunId/gpu-train.log"
      }
      return
    }

    $state = Get-InstanceState -Id $InstanceId
    Write-Host "  instance=$state (no manifest yet)"
    if ($state -in @("terminated", "shutting-down", "stopped", "stopping")) {
      throw "$Label instance ended without uploading a manifest. Check CloudWatch/SSM logs for $InstanceId."
    }

    Start-Sleep -Seconds 30
  }
}

$bucket = Get-StackOutput -Key "ArtifactsBucketName"
$launchTemplate = Get-StackOutput -Key "GpuLaunchTemplateName"
$runPrefix = if ($TrainingProfile -eq "gpuSmoke") { "gpu-smoke" } else { "gpu" }
$runId = "$runPrefix-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$sourceKey = "source/nnmcts-$runId.zip"
$launchedAt = (Get-Date).ToString("o")
$runType = if ($SmokeThenTrain) { "pipeline" } elseif ($TrainingProfile -eq "gpuSmoke") { "smoke" } else { "training" }
$runLabel = if ($SmokeThenTrain) { "GPU pipeline (smoke + training)" } elseif ($TrainingProfile -eq "gpuSmoke") { "GPU smoke test" } else { "GPU training" }

Write-Host "Packaging source..."
$zipPath = & (Join-Path $PSScriptRoot "package_source.ps1")

Write-Host "Uploading source to s3://$bucket/$sourceKey"
Invoke-AwsCli @("s3", "cp", $zipPath, "s3://$bucket/$sourceKey")

Write-Host "Launching $runLabel instance (max $($trainingConfig.MaxTrainingSeconds)s training / $($trainingConfig.MaxInstanceSeconds)s instance cap)..."
if ($SmokeThenTrain) {
  Write-Host "  smoke: gameType=$($smokeConfig.GameType) rounds=$($smokeConfig.Rounds) gamesPerRound=$($smokeConfig.GamesPerRound) epochs=$($smokeConfig.Epochs)"
}
Write-Host "  training: gameType=$($trainingConfig.GameType) rounds=$($trainingConfig.Rounds) gamesPerRound=$($trainingConfig.GamesPerRound) epochs=$($trainingConfig.Epochs) batchSize=$($trainingConfig.BatchSize) mctsIters=$($trainingConfig.MctsIters) selfPlayWorkers=$($trainingConfig.SelfPlayWorkers)"

$instanceTags = New-GpuTrainingTags -TrainingConfig $trainingConfig -Bucket $bucket -SourceKey $sourceKey -RunId $runId -RunType $runType -SmokeConfig $smokeConfig
$tagSpecifications = Format-Ec2TagSpecifications -Tags $instanceTags

$instanceJson = Invoke-AwsCli @(
  "ec2", "run-instances",
  "--launch-template", "LaunchTemplateName=$launchTemplate,Version=`$Latest",
  "--tag-specifications", $tagSpecifications,
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
  trainingProfile = $TrainingProfile
  runType = $runType
  smokeThenTrain = [bool]$SmokeThenTrain
  launchedAt = $launchedAt
  trainingConfig = $trainingConfig
}
if ($smokeConfig) {
  $metadata.smokeConfig = $smokeConfig
}
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($metadataPath, ($metadata | ConvertTo-Json -Depth 4), $utf8NoBom)

Write-Host ""
Write-Host "$runLabel launched."
Write-Host "  Instance ID:  $instanceId"
Write-Host "  Run ID:       $runId"
Write-Host "  Manifest:     s3://$bucket/runs/$runId/manifest.json"
Write-Host "  Run metadata: $metadataPath"
Write-Host ""

if ($Wait) {
  Wait-ForGpuRun -Bucket $bucket -RunId $runId -InstanceId $instanceId -Label $runLabel
  return
}

Write-Host "Check status and recent logs:"
Write-Host "  .\scripts\check_gpu_training.ps1 -MetadataPath `"$metadataPath`""
Write-Host ""
Write-Host "Poll until complete:"
Write-Host "  .\scripts\check_gpu_training.ps1 -MetadataPath `"$metadataPath`" -Follow"
