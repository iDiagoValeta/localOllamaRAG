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

pip install bert-score 2>/dev/null

echo "--> Limpiando librerías conflictivas..."
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
