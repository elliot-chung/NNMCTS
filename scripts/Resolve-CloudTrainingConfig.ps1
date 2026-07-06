function Resolve-CloudTrainingConfig {
  param(
    [ValidateSet("gpuSmoke", "gpu")]
    [string]$Profile = "gpu",
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
    [string]$PlayDevice,
    [string]$TrainDevice,
    [int]$MaxTrainingSeconds = 0,
    [int]$MaxInstanceSeconds = 0
  )

  $repoRoot = Split-Path -Parent $PSScriptRoot
  if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "config\cloud-training.json"
  }

  if (-not (Test-Path $ConfigPath)) {
    throw "Cloud training config not found: $ConfigPath"
  }

  $fileConfig = Get-Content $ConfigPath -Raw | ConvertFrom-Json
  if ($Profile -eq "gpu") {
    $training = $fileConfig.gpu
  }
  else {
    $training = if ($fileConfig.gpuSmoke) { $fileConfig.gpuSmoke } else { $fileConfig.smoke }
  }
  $timeouts = $fileConfig.timeouts

  $isSmoke = $Profile -eq "gpuSmoke"
  $defaultMaxTrainingSeconds = if ($isSmoke) { [int]$timeouts.maxSmokeTrainingSeconds } else { [int]$timeouts.maxTrainingSeconds }
  $defaultMaxInstanceSeconds = if ($isSmoke) { [int]$timeouts.maxSmokeInstanceSeconds } else { [int]$timeouts.maxInstanceSeconds }

  $resolved = [ordered]@{
    GameType = if ($GameType) { $GameType } else { $training.gameType }
    Rounds = if ($Rounds -gt 0) { $Rounds } else { [int]$training.rounds }
    GamesPerRound = if ($GamesPerRound -gt 0) { $GamesPerRound } else { [int]$training.gamesPerRound }
    Epochs = if ($Epochs -gt 0) { $Epochs } else { [int]$training.epochs }
    BatchSize = if ($BatchSize -gt 0) { $BatchSize } else { [int]$training.batchSize }
    MctsIters = if ($MctsIters -gt 0) { $MctsIters } else { [int]$training.mctsIters }
    Player1Type = if ($Player1Type) { $Player1Type } else { $training.player1Type }
    Player2Type = if ($Player2Type) { $Player2Type } else { $training.player2Type }
    SelfPlayWorkers = if ($SelfPlayWorkers -gt 0) { $SelfPlayWorkers } elseif ($training.PSObject.Properties.Name -contains "selfPlayWorkers") { [int]$training.selfPlayWorkers } else { 1 }
    PlayDevice = if ($PlayDevice) { $PlayDevice } elseif ($training.PSObject.Properties.Name -contains "playDevice") { [string]$training.playDevice } else { "cpu" }
    TrainDevice = if ($TrainDevice) { $TrainDevice } elseif ($training.PSObject.Properties.Name -contains "trainDevice") { [string]$training.trainDevice } else { "cuda" }
    NumEvalGames = if ($training.PSObject.Properties.Name -contains "numEvalGames") { [int]$training.numEvalGames } else { 0 }
    WinrateThreshold = if ($training.PSObject.Properties.Name -contains "winrateThreshold") { [double]$training.winrateThreshold } else { 0.55 }
    MaxTrainingSeconds = if ($MaxTrainingSeconds -gt 0) { $MaxTrainingSeconds } else { $defaultMaxTrainingSeconds }
    MaxInstanceSeconds = if ($MaxInstanceSeconds -gt 0) { $MaxInstanceSeconds } else { $defaultMaxInstanceSeconds }
    ConfigPath = $ConfigPath
    Profile = $Profile
  }

  return $resolved
}

function Add-TrainingConfigTags {
  param(
    [System.Collections.Generic.List[object]]$Tags,
    [hashtable]$TrainingConfig,
    [string]$Prefix = ""
  )

  $tagPrefix = if ($Prefix) { "$Prefix-" } else { "" }
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}game-type"; Value = [string]$TrainingConfig.GameType }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}rounds"; Value = [string]$TrainingConfig.Rounds }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}games-per-round"; Value = [string]$TrainingConfig.GamesPerRound }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}epochs"; Value = [string]$TrainingConfig.Epochs }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}batch-size"; Value = [string]$TrainingConfig.BatchSize }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}mcts-iters"; Value = [string]$TrainingConfig.MctsIters }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}player1-type"; Value = [string]$TrainingConfig.Player1Type }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}player2-type"; Value = [string]$TrainingConfig.Player2Type }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}self-play-workers"; Value = [string]$TrainingConfig.SelfPlayWorkers }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}play-device"; Value = [string]$TrainingConfig.PlayDevice }) | Out-Null
  $Tags.Add(@{ Key = "nnmcts-${tagPrefix}train-device"; Value = [string]$TrainingConfig.TrainDevice }) | Out-Null
  if ($TrainingConfig.ContainsKey("NumEvalGames") -and [int]$TrainingConfig.NumEvalGames -gt 0) {
    $Tags.Add(@{ Key = "nnmcts-${tagPrefix}num-eval-games"; Value = [string]$TrainingConfig.NumEvalGames }) | Out-Null
    $Tags.Add(@{ Key = "nnmcts-${tagPrefix}winrate-threshold"; Value = [string]$TrainingConfig.WinrateThreshold }) | Out-Null
  }
  if ($Prefix) {
    $Tags.Add(@{ Key = "nnmcts-${tagPrefix}max-training-seconds"; Value = [string]$TrainingConfig.MaxTrainingSeconds }) | Out-Null
  }
}

function New-GpuTrainingTags {
  param(
    [hashtable]$TrainingConfig,
    [string]$Bucket,
    [string]$SourceKey,
    [string]$RunId,
    [string]$RunType = "training",
    [hashtable]$SmokeConfig,
    [string]$InitialCheckpointName,
    [int]$StartRound = 0
  )

  $tags = [System.Collections.Generic.List[object]]::new()
  $tags.Add(@{ Key = "Name"; Value = "nnmcts-gpu-training" }) | Out-Null
  $tags.Add(@{ Key = "nnmcts-bucket"; Value = $Bucket }) | Out-Null
  $tags.Add(@{ Key = "nnmcts-source-key"; Value = $SourceKey }) | Out-Null
  $tags.Add(@{ Key = "nnmcts-run-id"; Value = $RunId }) | Out-Null
  $tags.Add(@{ Key = "nnmcts-run-type"; Value = $RunType }) | Out-Null
  $tags.Add(@{ Key = "nnmcts-max-training-seconds"; Value = [string]$TrainingConfig.MaxTrainingSeconds }) | Out-Null
  $tags.Add(@{ Key = "nnmcts-max-instance-seconds"; Value = [string]$TrainingConfig.MaxInstanceSeconds }) | Out-Null

  if ($InitialCheckpointName) {
    $tags.Add(@{ Key = "nnmcts-initial-checkpoint-name"; Value = [string]$InitialCheckpointName }) | Out-Null
  }
  if ($StartRound -gt 0) {
    $tags.Add(@{ Key = "nnmcts-start-round"; Value = [string]$StartRound }) | Out-Null
  }

  if ($SmokeConfig) {
    $tags.Add(@{ Key = "nnmcts-run-smoke"; Value = "true" }) | Out-Null
    Add-TrainingConfigTags -Tags $tags -TrainingConfig $SmokeConfig -Prefix "smoke"
  }

  Add-TrainingConfigTags -Tags $tags -TrainingConfig $TrainingConfig
  return $tags.ToArray()
}
