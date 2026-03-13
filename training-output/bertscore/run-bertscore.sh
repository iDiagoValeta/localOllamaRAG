#!/bin/bash

# =============================================================================
# SCRIPT DE EVALUACIÓN BERTSCORE EN SLURM (TFG)
# =============================================================================
# Lanzador de evaluación BERTScore para los 3 modelos en el clúster HPC.
# Evalúa modelos base y fine-tuneados sobre los test sets congelados.
#
# Uso:
#   sbatch run-bertscore.sh                    # Evalúa los 3 modelos
#   sbatch run-bertscore.sh gemma-3            # Solo un modelo
#   sbatch run-bertscore.sh llama-3            # Solo un modelo
# =============================================================================

#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=BERTScore
#SBATCH -o salida_bertscore_%j.log

echo "--> Nodo asignado: $(hostname)"
echo "--> Inicio: $(date)"

source ~/venv/TFG2526/bin/activate

# Instalar bert-score si no está disponible
pip install bert-score 2>/dev/null

echo "--> Limpiando librerías conflictivas..."
# triton: se desinstala DESPUÉS de bert-score para que torch no lo reinstale
# torchao 0.15.0: incompatible con torch 2.6.0+cu124 (crash al importar triton)
#   → no lo necesitamos (no usamos cuantización TorchAo en eval)
pip uninstall triton torchao -y 2>/dev/null

cd $SLURM_SUBMIT_DIR
echo "--> Directorio de trabajo: $(pwd)"

MODEL_ARG=""
if [ -n "$1" ]; then
    MODEL_ARG="--model $1"
    echo "--> Evaluando modelo: $1"
else
    echo "--> Evaluando los 3 modelos"
fi

python -u eval_bertscore.py $MODEL_ARG --output-dir ~/W

echo "--> Fin: $(date)"
