#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=0
#SBATCH --account=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=lonformer-repr-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="dasr8731@colorado.edu"

source /home/${USER}/.bashrc
source activate ./venv

mkdir -p metadata
mkdir -p models
mkdir -p logs
mkdir -p results
mkdir -p data

export TRANSFORMERS_CACHE=metadata/
export TORCH_HOME=./metadata/

python -m src.longformer