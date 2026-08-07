#!/bin/bash

#SBATCH --job-name=sshd
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j_%N.log
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "NODE: $(hostname)"
echo "PORT: $PORT"
echo "HOME: $HOME"

cd ~/flattened-leaves-processing
source .venv/bin/activate

python 03_crop.py


# start sshd on compute node
/usr/sbin/sshd -D -p ${PORT} \
  -f /dev/null \
  -h ${HOME}/.ssh/sshd_host_key \
  -o AuthorizedKeysFile=${HOME}/.ssh/authorized_keys
