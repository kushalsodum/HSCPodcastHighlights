#!/bin/bash

#SBATCH --job-name=train_linear
#SBATCH --account=eecs542f25_class
#SBATCH --mail-user=ksodum@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=01:00:00
#SBATCH --output=train_linear.log

echo "🚀 Starting GPT Model Evaluation..."

python train_rhapsody_linear.py

echo "✅ Evaluation completed!"
