#!/bin/bash

# =============================================================================
# SCRIPT DE EJECUCIÓN EN SLURM (TFG)
# =============================================================================
# Lanzador de entrenamiento para entorno HPC con 1 GPU.
# Define recursos, activa entorno Python del proyecto y ejecuta `train.py`
# con salida no bufferizada para seguimiento en tiempo real del log.
# =============================================================================

# =============================================================================
# SECCIÓN 1: RESERVA DE RECURSOS EN CLÚSTER
# =============================================================================
# Configuración de partición, GPU, CPU, memoria, tiempo máximo y archivo log.
# =============================================================================

#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --job-name=Qwen3RAG
#SBATCH -o salida_%j.log

# =============================================================================
# SECCIÓN 2: TRAZABILIDAD DE EJECUCIÓN
# =============================================================================
# Registra nodo asignado y marca temporal de inicio del trabajo.
# =============================================================================

echo "--> Nodo asignado: $(hostname)"
echo "--> Inicio: $(date)"

# =============================================================================
# SECCIÓN 3: ACTIVACIÓN DE ENTORNO DE PYTHON
# =============================================================================
# Activa el entorno virtual del TFG con dependencias de entrenamiento.
# =============================================================================

source ~/venv/TFG2526/bin/activate

# =============================================================================
# SECCIÓN 4: AJUSTE PREVIO DE DEPENDENCIAS
# =============================================================================
# Elimina `triton` si está presente para prevenir conflictos en este entorno.
# =============================================================================

echo "--> Limpiando librerías conflictivas..."
pip uninstall triton -y 2>/dev/null

# =============================================================================
# SECCIÓN 5: EJECUCIÓN DEL ENTRENAMIENTO
# =============================================================================
# Cambia al directorio de envío de SLURM y ejecuta entrenamiento principal.
# =============================================================================

cd $SLURM_SUBMIT_DIR
echo "--> Directorio de trabajo: $(pwd)"
python -u train.py
