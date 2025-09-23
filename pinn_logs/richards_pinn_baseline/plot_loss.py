# --- Training metrics quick-look (drop-in cell) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== Settings =====
file_path = Path(
    "/Users/jjason3/Downloads/Recharge project/pinn_logs/richards_pinn_baseline/training_metrics.csv"
)  # change if needed, e.g., Path("/mnt/data/training_metrics.csv")
rolling = 0  # rolling window for smoothing (set 0 to disable)

# ===== Load =====
df = pd.read_csv(file_path)
if "epoch" not in df.columns:
    # try to fabricate an epoch if missing
    df["epoch"] = np.arange(len(df))


def maybe_roll(s):
    if rolling and len(s) >= rolling:
        return s.rolling(rolling, min_periods=1, center=False).mean()
    return s


# ====== 1) Losses ======
loss_cols = [
    c
    for c in ["total_loss", "pde_loss", "surf_loss", "ic_h_loss", "ic_zb_loss"]
    if c in df.columns
]
plt.figure(figsize=(10, 6))
for c in loss_cols:
    plt.plot(df["epoch"], maybe_roll(df[c]), label=c)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (log)")
plt.title("Losses over epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ====== 2) Gradients ======
grad_cols = [
    c
    for c in ["total_grad", "pde_grad", "surf_grad", "ic_h_grad", "ic_zb_grad"]
    if c in df.columns
]
if grad_cols:
    plt.figure(figsize=(10, 6))
    for c in grad_cols:
        plt.plot(df["epoch"], maybe_roll(df[c]), label=c)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient (log)")
    plt.title("Gradient magnitudes over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ====== 3) Weights ======
wt_cols = [
    c
    for c in ["pde_weight", "surf_weight", "ic_h_weight", "ic_zb_weight"]
    if c in df.columns
]
if wt_cols:
    plt.figure(figsize=(10, 6))
    for c in wt_cols:
        plt.plot(df["epoch"], df[c], label=c)
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.title("Loss weights over epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ====== 4) LR & Cache Residuals ======
has_lr = "learning_rate" in df.columns
cache_cols = [
    c
    for c in ["cache_mean_residual", "cache_max_residual", "cache_std_residual"]
    if c in df.columns
]

if has_lr or cache_cols:
    plt.figure(figsize=(10, 6))
    if has_lr:
        plt.plot(df["epoch"], maybe_roll(df["learning_rate"]), label="learning_rate")
    for c in cache_cols:
        y = maybe_roll(df[c])
        plt.plot(df["epoch"], y, label=c)
    # Use log if residuals exist and are positive
    if cache_cols:
        ymin = np.nanmin([df[c].replace(0, np.nan).min() for c in cache_cols])
        if pd.notna(ymin) and ymin > 0:
            plt.yscale("log")
            plt.ylabel("Value (log)")
        else:
            plt.ylabel("Value")
    else:
        plt.ylabel("Value")
    plt.xlabel("Epoch")
    plt.title("Learning rate & cache residuals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print("Columns found:", list(df.columns))
print("Tip: adjust `rolling` for more/less smoothing.")
