param(
  [string]$Profile,
  [string]$Region,
  [string]$StackName,
  [string]$ConfigPath,
  [string]$GameType,
  [int]$Rounds = 0,
  [int]$GamesPerRound = 0,
  [int]$Epochs = 0,
  [int]$BatchSize = 0,
  [int]$MctsIters = 0,
  [string]$Player1Type,
  [string]$Player2Type,
  [int]$MaxTrainingSeconds = 0,
  [int]$MaxInstanceSeconds = 0
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region -StackName $StackName
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

. (Join-Path $PSScriptRoot "Resolve-CloudTrainingConfig.ps1")
$trainingConfig = Resolve-CloudTrainingConfig -Profile gpu -ConfigPath $ConfigPath -GameType $GameType -Rounds $Rounds -GamesPerRound $GamesPerRound -Epochs $Epochs -BatchSize $BatchSize -MctsIters $MctsIters -Player1Type $Player1Type -Player2Type $Player2Type -MaxTrainingSeconds $MaxTrainingSeconds -MaxInstanceSeconds $MaxInstanceSeconds

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

$bucket = Get-StackOutput -Key "ArtifactsBucketName"
$launchTemplate = Get-StackOutput -Key "GpuLaunchTemplateName"
$runId = "gpu-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$sourceKey = "source/nnmcts-$runId.zip"
$launchedAt = (Get-Date).ToString("o")

Write-Host "Packaging source..."
$zipPath = & (Join-Path $PSScriptRoot "package_source.ps1")

Write-Host "Uploading source to s3://$bucket/$sourceKey"
Invoke-AwsCli @("s3", "cp", $zipPath, "s3://$bucket/$sourceKey")

Write-Host "Launching GPU training instance (max $($trainingConfig.MaxTrainingSeconds)s training / $($trainingConfig.MaxInstanceSeconds)s instance cap)..."
Write-Host "  gameType=$($trainingConfig.GameType) rounds=$($trainingConfig.Rounds) gamesPerRound=$($trainingConfig.GamesPerRound) epochs=$($trainingConfig.Epochs) batchSize=$($trainingConfig.BatchSize) mctsIters=$($trainingConfig.MctsIters)"

$instanceTags = New-GpuTrainingTags -TrainingConfig $trainingConfig -Bucket $bucket -SourceKey $sourceKey -RunId $runId
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
  launchedAt = $launchedAt
  trainingConfig = $trainingConfig
}
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($metadataPath, ($metadata | ConvertTo-Json -Depth 4), $utf8NoBom)

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
