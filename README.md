# SMAP Recharge Flux - Physics-Informed Neural Network

Physics-Informed Neural Network (PINN) for solving Richards equation with moving water table boundary.

## Project Structure

```
SMAP-Recharge-Flux/
├── src/                          # Core source code
│   ├── pinn_models.py           # Neural network models (PressureHeadNet, WaterTableNet, RichardsPINN)
│   ├── train_loop.py            # Main training loop with GPU & checkpointing support
│   ├── training_utils.py        # Training utilities (sampling, losses, weight management)
│   ├── training_logger.py       # Training metrics logging
│   ├── normalization_helper.py  # Dimensionless normalization
│   ├── boundary_sampling.py     # Adaptive boundary sampling
│   ├── spike_detection.py       # Spike event detection
│   ├── surf_flux.py            # Surface flux data generation
│   └── visualization.py         # Plotting and visualization
│
├── hpc/                          # HPC cluster deployment
│   ├── train_script.py          # Standalone training script for HPC
│   ├── pace_job.sh             # SLURM job submission script (Georgia Tech PACE)
│   └── HPC_USAGE.md            # Complete HPC usage guide
│
├── notebooks/                    # Jupyter notebooks
│   └── forward_pinn.ipynb       # Interactive training notebook
│
├── data/                         # Input data
├── checkpoints/                  # Training checkpoints (created automatically)
├── pinn_logs/                    # Training logs
└── README.md                     # This file
```

## Quick Start

### Local Development (Jupyter Notebook)

```python
# In notebooks/forward_pinn.ipynb
import sys
sys.path.insert(0, '..')

from src.train_loop import train_pinn_pool_batch_autoweight

model = train_pinn_pool_batch_autoweight(
    # ... your parameters ...
    checkpoint_freq=5000,  # Enable checkpointing
)
```

### HPC Cluster (Georgia Tech PACE)

```bash
# Submit job to PACE
sbatch hpc/pace_job.sh

# Resume from checkpoint
python hpc/train_script.py --resume checkpoints/checkpoint_epoch_50000.pt
```

See [`hpc/HPC_USAGE.md`](hpc/HPC_USAGE.md) for complete HPC documentation.

## Features

### GPU Optimization
- ✅ Vectorized operations (100-1000x faster interpolation)
- ✅ Multi-GPU support (automatic DataParallel)
- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation

### Fault Tolerance
- ✅ Automatic checkpointing
- ✅ Resume from any checkpoint
- ✅ Checkpoint management (auto-cleanup)
- ✅ Complete state preservation

### Training Features
- ✅ Adaptive boundary sampling
- ✅ Spike event detection
- ✅ Adaptive loss weighting
- ✅ Cache pool sampling
- ✅ Comprehensive logging

## Requirements

```bash
pip install torch scipy numpy matplotlib
```

See `hpc/HPC_USAGE.md` for HPC-specific setup.

## Usage

### Training with Checkpointing

```python
from src.train_loop import train_pinn_pool_batch_autoweight

model = train_pinn_pool_batch_autoweight(
    soil_params=soil_params,
    q0_data=q0_data,
    # ... other parameters ...
    checkpoint_freq=5000,              # Save every 5000 epochs
    checkpoint_dir='checkpoints',
    keep_last_n_checkpoints=3,
)
```

### Resume from Checkpoint

```python
model = train_pinn_pool_batch_autoweight(
    # ... parameters ...
    resume_from_checkpoint='checkpoints/checkpoint_epoch_50000.pt',
)
```

### Enable GPU Optimizations

```python
model = train_pinn_pool_batch_autoweight(
    # ... parameters ...
    device='cuda',
    use_multi_gpu=True,        # Auto-detect multiple GPUs
    use_amp=True,              # Mixed precision (saves memory)
    grad_accumulation_steps=4, # Larger effective batch size
)
```

## Citation

If you use this code, please cite:

```
[Add your citation here]
```

## License

[Add your license here]
