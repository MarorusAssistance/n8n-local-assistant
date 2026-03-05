param(
  [switch]$Quick,
  [string]$Cases = "bench/cases.yaml",
  [string]$Experiments = "bench/experiments.yaml",
  [string]$Out = "bench/results"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ($Quick) {
  python -m bench run --quick --cases $Cases --experiments $Experiments --out $Out
} else {
  python -m bench run --cases $Cases --experiments $Experiments --out $Out
}
