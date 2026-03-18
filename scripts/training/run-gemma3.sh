#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --job-name=Gemma3RAG
#SBATCH -o salida_%j.log

echo "--> Nodo asignado: $(hostname)"
echo "--> Inicio: $(date)"

source ~/venv/TFG2526/bin/activate

echo "--> Limpiando librerías conflictivas..."
pip uninstall triton -y 2>/dev/null

cd $SLURM_SUBMIT_DIR
echo "--> Directorio de trabajo: $(pwd)"
python -u train-gemma3.py
