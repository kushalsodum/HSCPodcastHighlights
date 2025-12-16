#!/bin/bash

#SBATCH --job-name=predict
#SBATCH --account=eecs542f25_class
#SBATCH --mail-user=ksodum@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=00:10:00
#SBATCH --output=predict.log


echo "🚀 Starting Model Evaluation..."

python predict_rhapsody_bilstm.py

echo "✅ Evaluation completed!"
