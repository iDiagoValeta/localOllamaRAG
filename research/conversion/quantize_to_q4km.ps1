# Quantize a GGUF F16 checkpoint to Q4_K_M for Ollama / llama.cpp inference.
#
# Usage:
#   .\quantize_to_q4km.ps1 [<input.gguf>] [<output.gguf>]
#   Defaults: research\models\gguf\qwen-3\Qwen3-14B-f16.gguf ->
#             research\models\gguf\qwen-3\Qwen3-14B-Q4_K_M.gguf
#
# Dependencies:
#   - llama-quantize.exe from a local llama.cpp build, or extracted under
#     repo-root llama-bin\ (script may download a Windows CPU release if missing).

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  CONFIGURATION
#  +-- 1. Paths and default input/output GGUF
#
#  TOOLING
#  +-- 2. Resolve llama-quantize.exe (build + llama-bin)
#  +-- 3. Optional download of prebuilt llama.cpp binaries
#
#  PIPELINE
#  +-- 4. Run quantization and validate exit code
#
# ─────────────────────────────────────────────

$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────
# SECTION 1: PATHS AND DEFAULT INPUT/OUTPUT
# ─────────────────────────────────────────────

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)
$GGUF_DIR = Join-Path $PROJECT_ROOT "research\models\gguf"

$INPUT_GGUF = if ($args[0]) { $args[0] } else { Join-Path $GGUF_DIR "qwen-3\Qwen3-14B-f16.gguf" }
$OUTPUT_GGUF = if ($args[1]) { $args[1] } else { Join-Path $GGUF_DIR "qwen-3\Qwen3-14B-Q4_K_M.gguf" }

if (-not (Test-Path $INPUT_GGUF)) {
    Write-Error "No existe el archivo: $INPUT_GGUF"
    exit 1
}

# ─────────────────────────────────────────────
# SECTION 2: RESOLVE llama-quantize.exe
# ─────────────────────────────────────────────

$LLAMA_CPP = Join-Path $PROJECT_ROOT "llama.cpp"
$QUANTIZE_EXE = $null

$candidates = @(
    (Join-Path $LLAMA_CPP "build\bin\llama-quantize.exe"),
    (Join-Path $LLAMA_CPP "build\bin\Release\llama-quantize.exe"),
    (Join-Path $LLAMA_CPP "build\bin\RelWithDebInfo\llama-quantize.exe"),
    (Join-Path $PROJECT_ROOT "llama-bin\bin\llama-quantize.exe")
)

foreach ($c in $candidates) {
    if (Test-Path $c) {
        $QUANTIZE_EXE = $c
        break
    }
}

# ─────────────────────────────────────────────
# SECTION 3: DOWNLOAD PREBUILT BINARIES (FALLBACK)
# ─────────────────────────────────────────────

if (-not $QUANTIZE_EXE) {
    $BIN_DIR = Join-Path $PROJECT_ROOT "llama-bin"
    $QUANTIZE_EXE = Join-Path $BIN_DIR "bin\llama-quantize.exe"

    if (-not (Test-Path $QUANTIZE_EXE)) {
        Write-Host "Descargando llama.cpp binarios (Windows x64 CPU)..."
        $ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b7999/llama-b7999-bin-win-cpu-x64.zip"
        $ZIP_PATH = Join-Path $env:TEMP "llama-bin.zip"

        try {
            Invoke-WebRequest -Uri $ZIP_URL -OutFile $ZIP_PATH -UseBasicParsing
        } catch {
            Write-Error "Error descargando: $_"
            Write-Host "Descarga manual: $ZIP_URL"
            Write-Host "Extrae en: $BIN_DIR"
            exit 1
        }

        if (-not (Test-Path $BIN_DIR)) { New-Item -ItemType Directory -Path $BIN_DIR -Force | Out-Null }
        Expand-Archive -Path $ZIP_PATH -DestinationPath $BIN_DIR -Force
        Remove-Item $ZIP_PATH -Force -ErrorAction SilentlyContinue

        $found = Get-ChildItem $BIN_DIR -Recurse -Filter "llama-quantize.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            $QUANTIZE_EXE = $found.FullName
        } else {
            $QUANTIZE_EXE = Join-Path $BIN_DIR "bin\llama-quantize.exe"
        }
    }
}

if (-not (Test-Path $QUANTIZE_EXE)) {
    Write-Error "No se encontró llama-quantize. Compila llama.cpp o extrae el release en llama-bin/"
    exit 1
}

# ─────────────────────────────────────────────
# SECTION 4: RUN QUANTIZATION
# ─────────────────────────────────────────────

Write-Host "Convirtiendo a Q4_K_M..."
Write-Host "  Entrada:  $INPUT_GGUF"
Write-Host "  Salida:   $OUTPUT_GGUF"

& $QUANTIZE_EXE $INPUT_GGUF $OUTPUT_GGUF Q4_K_M

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error en la conversión (código $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Conversión completada: $OUTPUT_GGUF"
