#!/bin/bash
#SBATCH --job-name=train-qwen-3-5-4b-vision
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_qwen_3_5_4b_vision_%j.log
#SBATCH --error=logs/train_qwen_3_5_4b_vision_%j.err


# This is a SLURM script to train the Qwen 3.5 4B Vision model using 
# the main.py script in the qwen_3_5_4B_vision directory. This script is 
# suited to run on Snellius cluster, and assumes that 
# the Hugging Face cache directory is set to a location in the scratch space.

# set HF_HOME to be in the scratch space on the Snellius cluster
export HF_HOME=/scratch-shared/$USER/huggingface


# load required modules
module load 2025
module load CUDA/12.8.0
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# paths
PROJECT_DIR=$HOME/projects/fine-tuning/unsloth_ft
VENV_DIR=$PROJECT_DIR/.venv

# Create logs directory if it doesn't exist
mkdir -p $PROJECT_DIR/logs

# Activate virtual environment
source $VENV_DIR/bin/activate
# install packages
cd $PROJECT_DIR




# Print environment info for debugging
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python version: $(python --version)"
echo "HF_HOME: $HF_HOME"
echo "=========================================="

# run
python qwen_3_5_4B_vision/main.py --save=False --push=True

# Print completion info
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
