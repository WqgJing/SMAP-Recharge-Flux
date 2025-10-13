# HPC Training Scripts

This directory contains scripts for running PINN training on HPC clusters.

## Files

- **`train_forward_pinn.py`**: Main training script (converted from `forward_pinn.ipynb`)
- **`run_training.slurm`**: SLURM batch script for job submission
- **`README.md`**: This file

## Quick Start

### 1. Setup on HPC

```bash
# Clone/upload your repository to HPC
cd /path/to/SMAP-Recharge-Flux

# Make sure data directory exists
ls data/CalhounPrecipWY2015.csv

# Create logs directory
mkdir -p hpc/logs
```

### 2. Adjust SLURM Script

Edit `run_training.slurm` for your HPC environment:

```bash
#SBATCH --partition=gpu        # Change to your GPU partition name
#SBATCH --gres=gpu:1           # Request GPU (adjust for your system)
#SBATCH --mem=16G              # Adjust memory if needed
```

Uncomment and adjust module loads:
```bash
# module load python/3.9       # Your Python version
# module load cuda/11.8        # Your CUDA version
# module load cudnn/8.6        # Your cuDNN version
```

### 3. Submit Job

```bash
cd hpc
sbatch run_training.slurm
```

### 4. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output (real-time)
tail -f logs/train_<JOB_ID>.out

# View errors
tail -f logs/train_<JOB_ID>.err
```

## Training Configuration

Current settings (optimized for adaptive sampling):

```python
n_epochs=40000
cache_size=10000
batch_size=500
resample_freq=100              # Optimized: more responsive
high_residual_ratio=0.6        # Optimized: stronger focus
temperature=0.3                # Optimized: sharper concentration
```

### For Multi-GPU Training

Edit `train_forward_pinn.py` to enable multi-GPU:

```python
model = train_pinn_pool_batch_autoweight(
    ...
    use_multi_gpu=True,         # Enable DataParallel
    device='cuda',
)
```

And update SLURM script:
```bash
#SBATCH --gres=gpu:4           # Request 4 GPUs
```

## Output Files

Training generates:
- **Checkpoints**: `checkpoints_train/checkpoint_epoch_*.pt`
- **Final checkpoint**: `checkpoints_train/checkpoint_final.pt`
- **Logs**: `hpc/logs/train_<JOB_ID>.out`

## Troubleshooting

### Out of Memory
Reduce batch size or cache size:
```python
cache_size=5000
batch_size=250
```

### Job Timeout
Increase time limit or use checkpointing:
```bash
#SBATCH --time=48:00:00        # 48 hours
```

Resume from checkpoint:
```python
model = train_pinn_pool_batch_autoweight(
    ...
    resume_from_checkpoint='checkpoints_train/checkpoint_epoch_20000.pt',
)
```

### GPU Not Found
Check CUDA availability:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If False, check module loads or use CPU:
```python
device='cpu'  # Fallback to CPU
```

## Performance Tips

1. **Use GPU**: ~10-50× faster than CPU
2. **Enable AMP**: Mixed precision for 30-50% speedup
   ```python
   use_amp=True,
   ```
3. **Gradient accumulation**: Simulate larger batch without memory increase
   ```python
   grad_accumulation_steps=4,  # Effective batch = 500 × 4 = 2000
   ```
4. **Larger batches on HPC**: If you have enough GPU memory
   ```python
   cache_size=60000,
   batch_size=4000,
   ```

## Contact

See main repository README for issues and questions.
