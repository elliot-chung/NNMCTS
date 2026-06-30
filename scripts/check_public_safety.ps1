param(
  [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$issues = @()

function Add-Issue {
  param([string]$Message)
  $script:issues += $Message
}

$patterns = @(
  @{ Name = "AWS account ID"; Pattern = "\b\d{12}\b"; Allow = @("000000000000") },
  @{ Name = "Email address"; Pattern = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"; Allow = @() },
  @{ Name = "Windows user path"; Pattern = "C:\\Users\\[^\\]+\\"; Allow = @() }
)

$skipDirs = @(
  ".git", ".venv", "venv", "node_modules", "cdk.out", "dist", "smoke_local", "__pycache__"
)

$skipFiles = @(
  "infra\cdk.context.json"
)

Get-ChildItem -Path $Root -Recurse -File | Where-Object {
  $relative = $_.FullName.Substring($Root.Length + 1)
  if ($skipFiles -contains $relative) { return $false }
  -not ($skipDirs | Where-Object { $relative -like "$_*" -or $relative -like "*\$_\*" })
} | ForEach-Object {
  $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
  if (-not $content) { return }

  foreach ($rule in $patterns) {
    $matches = [regex]::Matches($content, $rule.Pattern)
    foreach ($match in $matches) {
      if ($rule.Allow -contains $match.Value) { continue }
      Add-Issue "$($rule.Name) in $($_.FullName): $($match.Value)"
    }
  }
}

if ($issues.Count -eq 0) {
  Write-Host "No obvious personal or account-specific leaks found."
  exit 0
}

Write-Host "Potential issues found:"
$issues | ForEach-Object { Write-Host "  - $_" }
exit 1
