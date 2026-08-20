#!/bin/bash

#SBATCH --job-name=sshd
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j_%N.log
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=24

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "NODE: $(hostname)"
echo "PORT: $PORT"
echo "HOME: $HOME"

cd ~/flattened-leaves-processing
source .venv/bin/activate

python 06_metrics.py
