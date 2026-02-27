# =============================================================================
# SCRIPT DE CUANTIZACIÓN GGUF (TFG)
# =============================================================================
# Convierte un modelo GGUF en precisión F16 a formato cuantizado Q4_K_M para
# inferencia eficiente en Ollama/llama.cpp.
#
# Uso:
#   .\quantize_to_q4km.ps1 [ruta_input.gguf] [ruta_output.gguf]
#
# Comportamiento por defecto:
#   models/gguf-output/Llama-3.1-8B-Teacher-f16.gguf
#   -> models/gguf-output/Llama-3.1-8B-Teacher-Q4_K_M.gguf
# =============================================================================

$ErrorActionPreference = "Stop"

# =============================================================================
# SECCIÓN 1: RUTAS Y PARÁMETROS DE ENTRADA
# =============================================================================
# Calcula rutas base del proyecto y resuelve archivo de entrada/salida a partir
# de argumentos opcionales o valores por defecto.
# =============================================================================

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)
$GGUF_DIR = Join-Path $PROJECT_ROOT "models\gguf-output"

$INPUT_GGUF = if ($args[0]) { $args[0] } else { Join-Path $GGUF_DIR "qwen-3\Qwen3-14B-Teacher-f16.gguf" }
$OUTPUT_GGUF = if ($args[1]) { $args[1] } else { Join-Path $GGUF_DIR "qwen-3\Qwen3-14B-Teacher-Q4_K_M.gguf" }

if (-not (Test-Path $INPUT_GGUF)) {
    Write-Error "No existe el archivo: $INPUT_GGUF"
    exit 1
}

# =============================================================================
# SECCIÓN 2: LOCALIZACIÓN DE HERRAMIENTA DE CUANTIZACIÓN
# =============================================================================
# Busca `llama-quantize.exe` en rutas esperadas del proyecto para reutilizar
# compilaciones locales o binarios previamente descargados.
# =============================================================================

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

# =============================================================================
# SECCIÓN 3: DESCARGA AUTOMÁTICA DE BINARIOS (FALLBACK)
# =============================================================================
# Si no hay binario local, descarga una release precompilada de llama.cpp,
# extrae su contenido y vuelve a resolver la ruta del ejecutable.
# =============================================================================

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

# =============================================================================
# SECCIÓN 4: EJECUCIÓN DE CUANTIZACIÓN Y SALIDA
# =============================================================================
# Ejecuta la conversión a Q4_K_M y valida código de retorno para asegurar que
# el artefacto final quede listo para registro en Ollama.
# =============================================================================

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
