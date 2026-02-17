@echo off
REM =============================================================================
REM SCRIPT ORQUESTADOR DE CONVERSIÓN Y DESPLIEGUE (TFG)
REM =============================================================================
REM Pipeline completo:
REM   1) Fusión LoRA + modelo base.
REM   2) Conversión a GGUF F16.
REM   3) Cuantización a Q4_K_M.
REM   4) Registro del modelo en Ollama.
REM
REM Uso:
REM   build_ollama.bat
REM
REM Requisitos:
REM   - Entorno conda `lora-gguf` disponible.
REM   - Repositorio `llama.cpp` accesible en la raíz del proyecto.
REM   - Script `quantize_to_q4km.ps1` en scripts/conversion.
REM =============================================================================

echo ========================================
echo Pipeline: LoRA a Q4_K_M (Ollama)
echo ========================================
echo.

REM =============================================================================
REM SECCIÓN 1: ACTIVACIÓN DE ENTORNO Y RUTAS
REM =============================================================================
REM Inicializa entorno de ejecución y define directorios de trabajo.
REM =============================================================================

call conda activate lora-gguf
if %errorlevel% neq 0 (
    echo ERROR: No se pudo activar el entorno conda 'lora-gguf'
    pause
    exit /b 1
)

set PROJECT_ROOT=%~dp0..\..
set MERGED_PATH=%PROJECT_ROOT%\models\merged-model
set GGUF_PATH=%PROJECT_ROOT%\models\gguf-output

REM =============================================================================
REM SECCIÓN 2: FUSIÓN DEL ADAPTADOR LORA
REM =============================================================================
REM Genera el modelo consolidado en `models/merged-model`.
REM =============================================================================

echo [1/4] Fusionando LoRA con modelo base...
python "%PROJECT_ROOT%\scripts\conversion\merge_lora.py"
if %errorlevel% neq 0 (
    echo ERROR: Fallo en merge
    pause
    exit /b 1
)

REM =============================================================================
REM SECCIÓN 3: CONVERSIÓN A GGUF F16
REM =============================================================================
REM Exporta el modelo fusionado al formato intermedio GGUF en precisión F16.
REM =============================================================================

echo.
echo [2/4] Convirtiendo a GGUF F16...
if not exist "%GGUF_PATH%" mkdir "%GGUF_PATH%"
python "%PROJECT_ROOT%\llama.cpp\convert_hf_to_gguf.py" "%MERGED_PATH%" --outfile "%GGUF_PATH%\Llama-3.1-8B-Teacher-f16.gguf" --outtype f16
if %errorlevel% neq 0 (
    echo ERROR: Fallo en conversion GGUF F16
    pause
    exit /b 1
)

REM =============================================================================
REM SECCIÓN 4: CUANTIZACIÓN A Q4_K_M
REM =============================================================================
REM Ejecuta cuantización para optimizar tamaño y rendimiento en inferencia.
REM =============================================================================

echo.
echo [3/4] Cuantizando a Q4_K_M...
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\conversion\quantize_to_q4km.ps1"
if %errorlevel% neq 0 (
    echo ERROR: Fallo en cuantizacion
    pause
    exit /b 1
)

REM =============================================================================
REM SECCIÓN 5: REGISTRO DEL MODELO EN OLLAMA
REM =============================================================================
REM Crea/actualiza el modelo `teacher-q4km` a partir del Modelfile.
REM =============================================================================

echo.
echo [4/4] Creando modelo en Ollama...
cd /d "%GGUF_PATH%"
ollama create teacher-q4km -f Modelfile
if %errorlevel% neq 0 (
    echo ERROR: Fallo en ollama create
    pause
    exit /b 1
)

echo.
echo ========================================
echo Completado. Modelo 'teacher-q4km' en Ollama.
echo ========================================
pause
