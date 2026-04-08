#!/bin/bash
#SBATCH --job-name=kitti-yolo11
#SBATCH --account=<project>
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1,nvme:950
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# CSC Mahti — YOLO11 KITTI training
# Submit with: sbatch scripts/run_train.sh
# Monitor with: squeue -u $USER

set -euo pipefail

module load pytorch

# Set up virtual environment (first run only)
VENV_DIR="$PWD/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" --system-site-packages
    source "$VENV_DIR/bin/activate"
    pip install -e .
else
    source "$VENV_DIR/bin/activate"
fi

# Create log directory
mkdir -p logs

# Copy data to fast local NVMe to avoid shared filesystem bottleneck
echo "Copying dataset to local NVMe..."
cp -r data/kitti/processed "$LOCAL_SCRATCH/kitti_processed"
echo "Data copied to $LOCAL_SCRATCH/kitti_processed"

# Run training
srun python3 -m kitti_object_detection.train \
    --data-dir "$LOCAL_SCRATCH/kitti_processed" \
    --epochs 150 \
    --imgsz 640 \
    --batch -1 \
    --device 0 \
    --name kitti_yolo11m

echo "Training complete."
