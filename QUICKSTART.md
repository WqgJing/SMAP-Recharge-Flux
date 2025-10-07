# Quick Start Guide

## Directory Structure

Your project is now organized into three main directories:

```
SMAP-Recharge-Flux/
├── src/          # Core Python modules (models, training, utilities)
├── hpc/          # HPC deployment scripts (SLURM, training script)
├── notebooks/    # Jupyter notebooks for interactive development
```

---

## For Local Development (Jupyter Notebook)

### 1. Start Jupyter from the project root:
```bash
cd /path/to/SMAP-Recharge-Flux
jupyter notebook
```

### 2. Open the notebook:
```
notebooks/forward_pinn.ipynb
```

### 3. Run cells - imports are already configured!
The notebook automatically imports from `src/`:
```python
from src.train_loop import train_pinn_pool_batch_autoweight
from src.pinn_models import RichardsPINN
# etc...
```

### 4. Enable checkpointing (optional):
```python
model = train_pinn_pool_batch_autoweight(
    # ... your parameters ...
    checkpoint_freq=5000,              # Add this line
    checkpoint_dir='checkpoints',
)
```

---

## For HPC Cluster (PACE)

### 1. Transfer files to PACE:
```bash
# From your local machine
rsync -av SMAP-Recharge-Flux/ username@login-pace.gatech.edu:~/SMAP-Recharge-Flux/
```

### 2. SSH to PACE:
```bash
ssh username@login-pace.gatech.edu
cd ~/SMAP-Recharge-Flux
```

### 3. Setup environment (first time only):
```bash
module load anaconda3
conda create -n smap_recharge python=3.10
conda activate smap_recharge
pip install -r requirements.txt
```

### 4. Edit SLURM script:
```bash
nano hpc/pace_job.sh
# Update your email address
```

### 5. Submit job:
```bash
mkdir -p logs checkpoints
sbatch hpc/pace_job.sh
```

### 6. Monitor:
```bash
# Check status
squeue -u $USER

# View output
tail -f logs/output-JOBID.out
```

### 7. Resume from checkpoint (if job crashes):
```bash
# Edit hpc/pace_job.sh and uncomment:
# python hpc/train_script.py --resume checkpoints/checkpoint_epoch_50000.pt

sbatch hpc/pace_job.sh
```

---

## For Python Scripts (Command Line)

### Run directly with Python:
```bash
python hpc/train_script.py --checkpoint-freq 5000
```

### With GPU optimizations:
```bash
python hpc/train_script.py \
    --checkpoint-freq 5000 \
    --use-amp \
    --grad-accumulation 4
```

### Resume from checkpoint:
```bash
python hpc/train_script.py \
    --resume checkpoints/checkpoint_epoch_50000.pt
```

---

## Common Tasks

### View training progress:
- **Local**: Check Jupyter notebook outputs
- **PACE**: `tail -f logs/output-JOBID.out`

### Access checkpoints:
```
checkpoints/
├── checkpoint_epoch_5000.pt
├── checkpoint_epoch_10000.pt
├── checkpoint_epoch_15000.pt
└── checkpoint_final.pt
```

### Load a trained model:
```python
import torch
from src.pinn_models import RichardsPINN

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_final.pt')

# Create model and load weights
model = RichardsPINN(...)  # same parameters as training
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## File Locations

| What | Where |
|------|-------|
| Core code | `src/*.py` |
| HPC scripts | `hpc/train_script.py`, `hpc/pace_job.sh` |
| Jupyter notebook | `notebooks/forward_pinn.ipynb` |
| Input data | `data/` |
| Checkpoints | `checkpoints/` (created automatically) |
| Training logs | `logs/` (created by SLURM) |
| Documentation | `README.md`, `hpc/HPC_USAGE.md` |

---

## Need Help?

- **General usage**: See `README.md`
- **HPC deployment**: See `hpc/HPC_USAGE.md`
- **Code reference**: All modules in `src/` have docstrings
