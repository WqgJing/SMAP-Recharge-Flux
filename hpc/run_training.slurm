#!/bin/bash
#SBATCH --job-name=pinn_train              # Job name
#SBATCH --output=logs/train_%j.out         # Output file (%j = job ID)
#SBATCH --error=logs/train_%j.err          # Error file
#SBATCH --account=gts-rbras6-coda20        # Account
#SBATCH --partition=gpu-h200               # GPU partition (H200)
#SBATCH --gres=gpu:h200:1                  # Request 1 H200 GPU
#SBATCH --cpus-per-task=8                  # Number of CPU cores
#SBATCH --mem=32G                          # Memory per node
#SBATCH --time=24:00:00                    # Time limit (24 hours)
#SBATCH --ntasks=1                         # Number of tasks

# Initialize conda (required for conda activate to work)
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate /storage/scratch1/9/jjason3/conda_envs/Recharge_PINN

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
nvidia-smi

# Run the training script (unbuffered output for real-time monitoring)
# Pass run name as first argument (default: 'default_run')
RUN_NAME="${1:-default_run}"
echo "Run name: $RUN_NAME"
python3 -u train_forward_pinn.py "$RUN_NAME"

echo "End time: $(date)"