# Package a minimal "user" zip: rag (includes rag/web/) + docs, excluding research/ and RagBench PDF trees.
# Run from anywhere; resolves repo root from this script location.
param(
    [string]$OutputZip = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $OutputZip) {
    $OutputZip = Join-Path $RepoRoot "monkeygrab-user-bundle.zip"
}
$Stage = Join-Path $env:TEMP ("monkeygrab-user-" + [Guid]::NewGuid().ToString("n"))
New-Item -ItemType Directory -Path $Stage -Force | Out-Null
try {
    foreach ($item in @("README.md", "CLAUDE.md", "pytest.ini")) {
        Copy-Item -LiteralPath (Join-Path $RepoRoot $item) -Destination (Join-Path $Stage $item) -Force
    }
    New-Item -ItemType Directory -Path (Join-Path $Stage "rag") -Force | Out-Null
    robocopy (Join-Path $RepoRoot "rag") (Join-Path $Stage "rag") /E /XD "en_ragbench_dev" "en_ragbench_eval" "en_ragbench_visual" "vector_db" "debug_rag" "__pycache__" "web\frontend\node_modules" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit $LASTEXITCODE" }
    if (Test-Path $OutputZip) { Remove-Item $OutputZip -Force }
    Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $OutputZip
    Write-Host "Wrote $OutputZip"
}
finally {
    Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
}
