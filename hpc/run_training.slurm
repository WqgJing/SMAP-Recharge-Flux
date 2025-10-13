#!/bin/bash
#SBATCH --job-name=pinn_train          # Job name
#SBATCH --output=logs/train_%j.out     # Output file (%j = job ID)
#SBATCH --error=logs/train_%j.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --mem=16G                      # Memory per node
#SBATCH --time=24:00:00                # Time limit (24 hours)
#SBATCH --partition=gpu                # Partition (adjust for your HPC)
#SBATCH --gres=gpu:1                   # Request 1 GPU

# Load modules (adjust for your HPC environment)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
nvidia-smi

# Run the training script
python3 train_forward_pinn.py

echo "End time: $(date)"
