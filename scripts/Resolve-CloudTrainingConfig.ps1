function Resolve-CloudTrainingConfig {
  param(
    [ValidateSet("smoke", "gpu")]
    [string]$Profile = "smoke",
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
    [int]$MaxRuntimeSeconds = 0,
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
  $training = if ($Profile -eq "gpu") { $fileConfig.gpu } else { $fileConfig.smoke }
  $timeouts = $fileConfig.timeouts

  $resolved = [ordered]@{
    GameType = if ($GameType) { $GameType } else { $training.gameType }
    Rounds = if ($Rounds -gt 0) { $Rounds } else { [int]$training.rounds }
    GamesPerRound = if ($GamesPerRound -gt 0) { $GamesPerRound } else { [int]$training.gamesPerRound }
    Epochs = if ($Epochs -gt 0) { $Epochs } else { [int]$training.epochs }
    BatchSize = if ($BatchSize -gt 0) { $BatchSize } else { [int]$training.batchSize }
    MctsIters = if ($MctsIters -gt 0) { $MctsIters } else { [int]$training.mctsIters }
    Player1Type = if ($Player1Type) { $Player1Type } else { $training.player1Type }
    Player2Type = if ($Player2Type) { $Player2Type } else { $training.player2Type }
    SelfPlayWorkers = if ($SelfPlayWorkers -gt 0) { $SelfPlayWorkers } elseif ($Profile -eq "gpu" -and $training.PSObject.Properties.Name -contains "selfPlayWorkers") { [int]$training.selfPlayWorkers } else { 1 }
    MaxRuntimeSeconds = if ($MaxRuntimeSeconds -gt 0) { $MaxRuntimeSeconds } else { [int]$timeouts.maxRuntimeSeconds }
    MaxTrainingSeconds = if ($MaxTrainingSeconds -gt 0) { $MaxTrainingSeconds } else { [int]$timeouts.maxTrainingSeconds }
    MaxInstanceSeconds = if ($MaxInstanceSeconds -gt 0) { $MaxInstanceSeconds } else { [int]$timeouts.maxInstanceSeconds }
    CodeBuildTimeoutMinutes = [int]$timeouts.codeBuildTimeoutMinutes
    CodeBuildQueuedTimeoutMinutes = [int]$timeouts.codeBuildQueuedTimeoutMinutes
    ConfigPath = $ConfigPath
    Profile = $Profile
  }

  return $resolved
}

function New-CodeBuildEnvironmentOverrides {
  param(
    [hashtable]$TrainingConfig
  )

  $envMap = [ordered]@{
    GAME_TYPE = [string]$TrainingConfig.GameType
    ROUNDS = [string]$TrainingConfig.Rounds
    GAMES_PER_ROUND = [string]$TrainingConfig.GamesPerRound
    EPOCHS = [string]$TrainingConfig.Epochs
    BATCH_SIZE = [string]$TrainingConfig.BatchSize
    MCTS_ITERS = [string]$TrainingConfig.MctsIters
    PLAYER1_TYPE = [string]$TrainingConfig.Player1Type
    PLAYER2_TYPE = [string]$TrainingConfig.Player2Type
    SELF_PLAY_WORKERS = [string]$TrainingConfig.SelfPlayWorkers
    MAX_RUNTIME_SECONDS = [string]$TrainingConfig.MaxRuntimeSeconds
  }

  $overrides = @()
  foreach ($entry in $envMap.GetEnumerator()) {
    $overrides += "name=$($entry.Key),value=$($entry.Value),type=PLAINTEXT"
  }
  return $overrides
}

function New-GpuTrainingTags {
  param(
    [hashtable]$TrainingConfig,
    [string]$Bucket,
    [string]$SourceKey,
    [string]$RunId
  )

  return @(
    @{ Key = "Name"; Value = "nnmcts-gpu-training" },
    @{ Key = "nnmcts-bucket"; Value = $Bucket },
    @{ Key = "nnmcts-source-key"; Value = $SourceKey },
    @{ Key = "nnmcts-run-id"; Value = $RunId },
    @{ Key = "nnmcts-game-type"; Value = [string]$TrainingConfig.GameType },
    @{ Key = "nnmcts-rounds"; Value = [string]$TrainingConfig.Rounds },
    @{ Key = "nnmcts-games-per-round"; Value = [string]$TrainingConfig.GamesPerRound },
    @{ Key = "nnmcts-epochs"; Value = [string]$TrainingConfig.Epochs },
    @{ Key = "nnmcts-batch-size"; Value = [string]$TrainingConfig.BatchSize },
    @{ Key = "nnmcts-mcts-iters"; Value = [string]$TrainingConfig.MctsIters },
    @{ Key = "nnmcts-player1-type"; Value = [string]$TrainingConfig.Player1Type },
    @{ Key = "nnmcts-player2-type"; Value = [string]$TrainingConfig.Player2Type },
    @{ Key = "nnmcts-self-play-workers"; Value = [string]$TrainingConfig.SelfPlayWorkers },
    @{ Key = "nnmcts-max-training-seconds"; Value = [string]$TrainingConfig.MaxTrainingSeconds },
    @{ Key = "nnmcts-max-instance-seconds"; Value = [string]$TrainingConfig.MaxInstanceSeconds }
  )
}
