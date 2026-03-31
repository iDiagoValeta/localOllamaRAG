#!/bin/bash

#SBATCH -p docencia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=baseEval
#SBATCH -o salida_%j.log

echo "--> Nodo asignado: $(hostname)"
echo "--> Inicio: $(date)"

source ~/venv/TFG2526/bin/activate

echo "--> Limpiando librerías conflictivas..."
pip uninstall triton -y 2>/dev/null

cd $SLURM_SUBMIT_DIR
echo "--> Directorio de trabajo: $(pwd)"
python -u train.py
