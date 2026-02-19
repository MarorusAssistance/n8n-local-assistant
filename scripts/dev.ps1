param(
  [string]$OpenWebUiDir = $env:OPENWEBUI_DIR,
  [string]$OpenWebUiHost = "0.0.0.0",
  [int]$OpenWebUiPort = 8080,
  [switch]$NoLmStudio,
  [string]$LmStudioCli = "lms",
  [string]$LmStudioBind = "0.0.0.0",
  [int]$LmStudioPort = 1234,
  [string]$LmStudioLlmModel = "",
  [string]$LmStudioEmbeddingModel = "",
  [int]$LmStudioLlmContextLength = 10000,
  [int]$LmStudioEmbeddingContextLength = 4096,
  [string]$LmStudioGpuOffload = "max",
  [string]$LmStudioLlmIdentifier = "",
  [string]$LmStudioEmbeddingIdentifier = "",
  [switch]$LmStudioCors,
  [switch]$ForceLmStudioReload,
  [string]$ApiHost = "0.0.0.0",
  [int]$ApiPort = 8000,
  [switch]$NoReload
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OpenWebUiDir)) {
  $OpenWebUiDir = "C:\Users\maror\Projects\openwebui"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$openWebUiExe = Join-Path $OpenWebUiDir ".venv\Scripts\open-webui.exe"
$uvicornExe = Join-Path $repoRoot ".venv\Scripts\uvicorn.exe"

$envFile = Join-Path $repoRoot ".env"

function Get-DotEnvMap {
  param([string]$Path)

  $map = @{}
  if (!(Test-Path $Path)) {
    return $map
  }

  foreach ($line in Get-Content $Path) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
      continue
    }

    $parts = $trimmed.Split("=", 2)
    if ($parts.Count -ne 2) {
      continue
    }

    $key = $parts[0].Trim()
    $value = $parts[1].Trim()
    if ([string]::IsNullOrWhiteSpace($key)) {
      continue
    }

    if ((($value.StartsWith(""")) -and ($value.EndsWith("""))) -or (($value.StartsWith("'")) -and ($value.EndsWith("'")))) {
      if ($value.Length -ge 2) {
        $value = $value.Substring(1, $value.Length - 2)
      }
    }

    $map[$key] = $value
  }

  return $map
}

$dotEnv = Get-DotEnvMap -Path $envFile

if ([string]::IsNullOrWhiteSpace($LmStudioLlmModel)) {
  $LmStudioLlmModel = $dotEnv["LLM_MODEL"]
}
if ([string]::IsNullOrWhiteSpace($LmStudioEmbeddingModel)) {
  $LmStudioEmbeddingModel = $dotEnv["EMBEDDING_MODEL"]
}

if ([string]::IsNullOrWhiteSpace($LmStudioLlmModel)) {
  $LmStudioLlmModel = "mistralai/ministral-3-14b-reasoning"
}
if ([string]::IsNullOrWhiteSpace($LmStudioEmbeddingModel)) {
  $LmStudioEmbeddingModel = "text-embedding-bge-m3"
}

if ([string]::IsNullOrWhiteSpace($LmStudioLlmIdentifier)) {
  $LmStudioLlmIdentifier = $LmStudioLlmModel
}
if ([string]::IsNullOrWhiteSpace($LmStudioEmbeddingIdentifier)) {
  $LmStudioEmbeddingIdentifier = $LmStudioEmbeddingModel
}

if (!(Test-Path $openWebUiExe)) {
  Write-Error "Open WebUI not found at $openWebUiExe. Set OPENWEBUI_DIR or pass -OpenWebUiDir."
  exit 1
}

if (!(Test-Path $uvicornExe)) {
  Write-Error "uvicorn not found at $uvicornExe. Create the venv in $repoRoot."
  exit 1
}

if (-not $NoLmStudio) {
  $lmStudioCmd = Get-Command $LmStudioCli -ErrorAction SilentlyContinue
  if (-not $lmStudioCmd) {
    Write-Error "LM Studio CLI not found. Install LM Studio or set -LmStudioCli to the lms.exe path."
    exit 1
  }
  $lmStudioExe = $lmStudioCmd.Source

  $serverRunning = $false
  try {
    $statusText = & $lmStudioExe server status 2>$null
    if ($statusText -match "server is running") {
      $serverRunning = $true
    }
  } catch {
    $serverRunning = $false
  }

  if (-not $serverRunning) {
    $serverArgs = @("server", "start", "--port", $LmStudioPort, "--bind", $LmStudioBind)
    if ($LmStudioCors) {
      $serverArgs += "--cors"
    }
    & $lmStudioExe @serverArgs | Out-Null
  }

  $loaded = @()
  try {
    $loadedJson = & $lmStudioExe ps --json --port $LmStudioPort 2>$null
    if ($loadedJson) {
      $loaded = $loadedJson | ConvertFrom-Json
    }
  } catch {
    $loaded = @()
  }

  function Ensure-LmStudioModelLoaded {
    param(
      [string]$Type,
      [string]$ModelKey,
      [string]$Identifier,
      [int]$ContextLength
    )

    $existing = $loaded | Where-Object {
      $_.type -eq $Type -and ($_.identifier -eq $Identifier -or $_.modelKey -eq $ModelKey)
    } | Select-Object -First 1

    if ($existing) {
      if ($ForceLmStudioReload -and $ContextLength -gt 0 -and $existing.contextLength -ne $ContextLength) {
        & $lmStudioExe unload $existing.identifier --port $LmStudioPort | Out-Null
      } else {
        return
      }
    }

    $loadArgs = @("load", $ModelKey, "--yes", "--port", $LmStudioPort, "--identifier", $Identifier)
    if ($LmStudioGpuOffload) {
      $loadArgs += @("--gpu", $LmStudioGpuOffload)
    }
    if ($ContextLength -gt 0) {
      $loadArgs += @("--context-length", $ContextLength)
    }
    & $lmStudioExe @loadArgs | Out-Null
  }

  Ensure-LmStudioModelLoaded -Type "embedding" -ModelKey $LmStudioEmbeddingModel -Identifier $LmStudioEmbeddingIdentifier -ContextLength $LmStudioEmbeddingContextLength
  Ensure-LmStudioModelLoaded -Type "llm" -ModelKey $LmStudioLlmModel -Identifier $LmStudioLlmIdentifier -ContextLength $LmStudioLlmContextLength
}

$openWebUiArgs = @("serve", "--host", $OpenWebUiHost, "--port", $OpenWebUiPort)
Start-Process -FilePath $openWebUiExe -ArgumentList $openWebUiArgs -WorkingDirectory $OpenWebUiDir

$uvicornArgs = @("app.main:app", "--host", $ApiHost, "--port", $ApiPort)
if (-not $NoReload) {
  $uvicornArgs += "--reload"
}

& $uvicornExe @uvicornArgs
