#!/bin/bash
#SBATCH --job-name=kitti-predict
#SBATCH --account=<project>
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

# CSC Mahti — YOLO11 KITTI inference
# Submit with: sbatch scripts/run_predict.sh

set -euo pipefail

module load pytorch

source .venv/bin/activate

srun python3 -m kitti_object_detection.predict \
    --device 0

echo "Inference complete."
