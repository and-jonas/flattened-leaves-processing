#!/bin/bash

#SBATCH --job-name=sshd
#SBATCH --partition=cpu
#SBATCH --array=1-5
#SBATCH --time=10:00:00
#SBATCH --output=%x_%A_%a_%N.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "NODE: $(hostname)"
echo "PORT: $PORT"
echo "HOME: $HOME"
echo "TASK: $SLURM_ARRAY_TASK_ID"

cd ~/flattened-leaves-processing
source .venv/bin/activate

echo "Running: python 05_align.py ${SLURM_ARRAY_TASK_ID}"
python 05_align.py $SLURM_ARRAY_TASK_ID

echo "Finished task ${SLURM_ARRAY_TASK_ID}"
