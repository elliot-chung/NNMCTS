param(
  [string]$Profile,
  [string]$Region,
  [switch]$DeployOnly,
  [switch]$RunOnly,
  [switch]$SmokeOnly,
  [switch]$SkipSmoke,
  [string]$ConfigPath,
  [string]$GameType,
  [int]$Rounds = 0,
  [int]$GamesPerRound = 0,
  [int]$Epochs = 0,
  [int]$BatchSize = 0,
  [int]$MctsIters = 0,
  [string]$Player1Type,
  [string]$Player2Type,
  [int]$SelfPlayWorkers = 0,
  [string]$InitialCheckpointPath,
  [int]$StartRound = 0,
  [int]$MaxTrainingSeconds = 0,
  [int]$MaxInstanceSeconds = 0,
  [switch]$Wait
)

. (Join-Path $PSScriptRoot "Resolve-AwsConfig.ps1")
$awsConfig = Resolve-AwsConfig -Profile $Profile -Region $Region
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
    if ($LASTEXITCODE -ne 0) {
      throw "CDK bootstrap failed for aws://$account/$Region"
    }
    npx cdk deploy $StackName --profile $Profile --require-approval never
    if ($LASTEXITCODE -ne 0) {
      throw "CDK deploy failed for $StackName. Fix infrastructure errors before launching training."
    }
  }
  finally {
    Pop-Location
  }
}

if ($DeployOnly) {
  Write-Host "Deploy complete (--DeployOnly)."
  exit 0
}

$launchArgs = @{
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
  SelfPlayWorkers = $SelfPlayWorkers
  InitialCheckpointPath = $InitialCheckpointPath
  StartRound = $StartRound
  MaxTrainingSeconds = $MaxTrainingSeconds
  MaxInstanceSeconds = $MaxInstanceSeconds
  Wait = $Wait.IsPresent
}

if ($SmokeOnly) {
  $launchArgs.TrainingProfile = "gpuSmoke"
  & (Join-Path $PSScriptRoot "run_gpu_training.ps1") @launchArgs
  exit $LASTEXITCODE
}

if ($SkipSmoke) {
  $launchArgs.TrainingProfile = "gpu"
  & (Join-Path $PSScriptRoot "run_gpu_training.ps1") @launchArgs
  exit $LASTEXITCODE
}

Write-Host "=== GPU pipeline (smoke test + full training on one instance) ==="
$launchArgs.SmokeThenTrain = $true
if (-not $launchArgs.Wait) {
  $launchArgs.Wait = $true
}
& (Join-Path $PSScriptRoot "run_gpu_training.ps1") @launchArgs
exit $LASTEXITCODE
