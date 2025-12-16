#!/bin/bash

#SBATCH --job-name=train_bilstm
#SBATCH --account=cse576f25s001_class
#SBATCH --mail-user=ksodum@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --time=02:00:00
#SBATCH --output=train_bilstm.log

# Script to evaluate SFT-trained GPT model on test questions
# This script runs the evaluation with default parameters

echo "🚀 Starting GPT Model Evaluation..."

python train_rhapsody_bilstm.py

echo "✅ Evaluation completed!"
