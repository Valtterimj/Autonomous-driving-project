#!/bin/bash
#SBATCH --job-name=kitti-eval
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# CSC Mahti — KITTI evaluation (CPU-only, no GPU needed)
# Submit with: sbatch scripts/run_evaluate.sh

set -euo pipefail

module load pytorch

source .venv/bin/activate

# Run KITTI AP evaluation
srun python3 -m kitti_object_detection.evaluate

# Generate visualizations
srun python3 -m kitti_object_detection.visualize --num-samples 30

echo "Evaluation and visualization complete."
