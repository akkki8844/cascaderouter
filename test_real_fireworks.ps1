# One-command harness-style test against the REAL Fireworks API.
# Prereqs: copy .env.example to .env and paste your real FIREWORKS_API_KEY.
# Runs harness_main.py exactly like the judging container does (env-injected
# key/base-url/models, /input -> /output contract) on the 8-category sample.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Write-Error "No .env found. Copy .env.example to .env and paste your FIREWORKS_API_KEY."
}

# Load .env into this process's environment
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$' -and $_ -notmatch '^\s*#') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
if ($env:FIREWORKS_API_KEY -like "*your_key_here*") {
    Write-Error "FIREWORKS_API_KEY in .env is still the placeholder."
}

$env:TASKS_INPUT = "eval\sample_input.json"
$env:RESULTS_OUTPUT = "harness_real_out.json"

Write-Host "Base URL : $env:FIREWORKS_BASE_URL"
Write-Host "Models   : $env:ALLOWED_MODELS"
Write-Host "Running harness_main.py on eval\sample_input.json ..."
python harness_main.py
if ($LASTEXITCODE -ne 0) { Write-Error "harness_main.py exited $LASTEXITCODE" }

Write-Host "`n--- results ---"
Get-Content $env:RESULTS_OUTPUT
Write-Host "`nDone. Inspect logs\decisions.jsonl-free routes should show 0 remote tokens;"
Write-Host "escalations should name a gemma-4-31b-it model id."
