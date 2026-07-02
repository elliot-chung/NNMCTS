param(
  [string]$Profile,
  [string]$Region,
  [switch]$DeployOnly,
  [switch]$RunOnly,
  [switch]$Gpu,
  [string]$ConfigPath,
  [string]$GameType,
  [int]$Rounds = 0,
  [int]$GamesPerRound = 0,
  [int]$Epochs = 0,
  [int]$BatchSize = 0,
  [int]$MctsIters = 0,
  [string]$Player1Type,
  [string]$Player2Type,
  [int]$MaxRuntimeSeconds = 0,
  [int]$MaxTrainingSeconds = 0,
  [int]$MaxInstanceSeconds = 0
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region
$Profile = $awsConfig.Profile
$Region = $awsConfig.Region
$StackName = $awsConfig.StackName

if ($Gpu -and -not $DeployOnly) {
  $gpuArgs = @{
    Profile = $Profile
    Region = $Region
    ConfigPath = $ConfigPath
    GameType = $GameType
    Rounds = $Rounds
    GamesPerRound = $GamesPerRound
    Epochs = $Epochs
    BatchSize = $BatchSize
    MctsIters = $MctsIters
    Player1Type = $Player1Type
    Player2Type = $Player2Type
    MaxTrainingSeconds = $MaxTrainingSeconds
    MaxInstanceSeconds = $MaxInstanceSeconds
  }
  & (Join-Path $PSScriptRoot "run_gpu_training.ps1") @gpuArgs
  exit $LASTEXITCODE
}

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

function Get-StackOutput {
  param([string]$Key)
  $json = Invoke-AwsCli @(
    "cloudformation", "describe-stacks",
    "--stack-name", $StackName,
    "--query", "Stacks[0].Outputs[?OutputKey=='$Key'].OutputValue",
    "--output", "text"
  )
  return $json.Trim()
}

if (-not $RunOnly) {
  Write-Host "Bootstrapping CDK (if needed)..."
  Push-Location $InfraDir
  try {
    Invoke-AwsCli @("sts", "get-caller-identity", "--query", "Account", "--output", "text") | Out-Null
    $account = (Invoke-AwsCli @("sts", "get-caller-identity", "--query", "Account", "--output", "text")).Trim()
    $env:CDK_DEFAULT_ACCOUNT = $account
    $env:CDK_DEFAULT_REGION = $Region

    if (-not (Test-Path "node_modules")) {
      Write-Host "Installing CDK dependencies..."
      npm install --silent
    }

    Write-Host "Deploying $StackName to $Region..."
    npx cdk bootstrap "aws://$account/$Region" --profile $Profile
    npx cdk deploy $StackName --profile $Profile --require-approval never
  }
  finally {
    Pop-Location
  }
}

if ($DeployOnly) {
  Write-Host "Deploy complete (--DeployOnly)."
  exit 0
}

. (Join-Path $PSScriptRoot "Resolve-CloudTrainingConfig.ps1")
$trainingConfig = Resolve-CloudTrainingConfig -Profile smoke -ConfigPath $ConfigPath -GameType $GameType -Rounds $Rounds -GamesPerRound $GamesPerRound -Epochs $Epochs -BatchSize $BatchSize -MctsIters $MctsIters -Player1Type $Player1Type -Player2Type $Player2Type -MaxRuntimeSeconds $MaxRuntimeSeconds

$bucket = Get-StackOutput -Key "ArtifactsBucketName"
$project = Get-StackOutput -Key "CodeBuildProjectName"
$sourceKey = "source/nnmcts-$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"

Write-Host "Packaging source from $RepoRoot..."
$zipPath = & (Join-Path $PSScriptRoot "package_source.ps1")

Write-Host "Uploading source to s3://$bucket/$sourceKey"
Invoke-AwsCli @("s3", "cp", $zipPath, "s3://$bucket/$sourceKey")

Write-Host "Starting CodeBuild project $project (max $($trainingConfig.MaxRuntimeSeconds)s)..."
Write-Host "  gameType=$($trainingConfig.GameType) rounds=$($trainingConfig.Rounds) gamesPerRound=$($trainingConfig.GamesPerRound) epochs=$($trainingConfig.Epochs) batchSize=$($trainingConfig.BatchSize) mctsIters=$($trainingConfig.MctsIters)"

$codeBuildCapSeconds = $trainingConfig.CodeBuildTimeoutMinutes * 60
if ($trainingConfig.MaxRuntimeSeconds -gt $codeBuildCapSeconds) {
  Write-Warning "MaxRuntimeSeconds ($($trainingConfig.MaxRuntimeSeconds)) exceeds the deployed CodeBuild project timeout ($codeBuildCapSeconds s). Redeploy after raising timeouts.codeBuildTimeoutMinutes in config/cloud-training.json, or lower -MaxRuntimeSeconds."
}

$envOverrides = New-CodeBuildEnvironmentOverrides -TrainingConfig $trainingConfig
$buildJson = Invoke-AwsCli (@(
  "codebuild", "start-build",
  "--project-name", $project,
  "--source-type-override", "S3",
  "--source-location-override", "$bucket/$sourceKey",
  "--environment-variables-override"
) + $envOverrides + @(
  "--output", "json"
)) | ConvertFrom-Json

$buildId = $buildJson.build.id
Write-Host "Build ID: $buildId"
Write-Host "Logs: /nnmcts/codebuild"
Write-Host "Waiting for build to finish..."

while ($true) {
  Start-Sleep -Seconds 20
  $statusJson = Invoke-AwsCli @(
    "codebuild", "batch-get-builds",
    "--ids", $buildId,
    "--output", "json"
  ) | ConvertFrom-Json

  $build = $statusJson.builds[0]
  $phase = $build.currentPhase
  $status = $build.buildStatus
  Write-Host "  phase=$phase status=$status"

  if ($status -in @("SUCCEEDED", "FAILED", "FAULT", "STOPPED", "TIMED_OUT")) {
    if ($status -ne "SUCCEEDED") {
      throw "CodeBuild finished with status $status. Check CloudWatch logs at /nnmcts/codebuild"
    }
    break
  }
}

$manifestUri = "s3://$bucket/runs/$buildId/manifest.json"
Write-Host "Build succeeded. Fetching manifest from $manifestUri"
Invoke-AwsCli @("s3", "cp", $manifestUri, "-") | Write-Host

Write-Host ""
Write-Host "Artifacts:"
Write-Host "  s3://$bucket/runs/$buildId/"
Write-Host "  Checkpoints: s3://$bucket/runs/$buildId/checkpoints/"
