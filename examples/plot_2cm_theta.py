import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Absolute path to the Excel file
EXCEL_PATH = "/Users/jasonjing/Documents/smap/SMAP-Recharge-Flux/data/2017_HalfHourly_UTC_ForestSite1.xlsx"

# Optional manual override: set to the exact column name if auto-detect fails
TARGET_THETA_COLUMN = None  # e.g., "Theta_2cm" or "SWC_2cm"

# Optional manual override for time column
TARGET_TIME_COLUMN = None   # e.g., "DateTime" or "Timestamp"

# Output path
FIG_PATH = Path("/Users/jasonjing/Documents/smap/SMAP-Recharge-Flux/figures/plot_2cm_theta.png")


def detect_time_column(df):
    """Try to find the datetime column by name hints and parsability."""
    name_hints = [c for c in df.columns if re.search(r"date|time|timestamp", str(c), re.IGNORECASE)]
    for col in name_hints:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True, infer_datetime_format=True)
        if parsed.notna().mean() > 0.8:
            return col
    # Fallback: evaluate all columns and pick the most parsable one
    best_col, best_score = None, 0.0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True, infer_datetime_format=True)
        score = float(parsed.notna().mean())
        if score > best_score:
            best_col, best_score = col, score
    return best_col if best_score > 0.5 else None


def detect_2cm_theta_column(df):
    """Heuristically find the 2 cm soil moisture/theta column by header text."""
    def norm(s):
        return str(s).lower().replace("_", " ").replace("-", " ").strip()

    # Common moisture/theta tokens and 2 cm depth tokens
    moisture_tokens = ["theta", "swc", "moist", "vwc", "soil water", "soil moisture", "sm"]
    depth_tokens = ["2 cm", "2cm", "0.02 m", "0.02m", "2 centimeter", "2 centimetre"]

    # Pass 1: both moisture and explicit 2cm tokens present
    for col in df.columns:
        n = norm(col)
        if any(tok in n for tok in moisture_tokens) and any(tok in n for tok in depth_tokens):
            return col

    # Pass 2: moisture token present + contains both '2' and 'cm' anywhere
    for col in df.columns:
        n = norm(col)
        if any(tok in n for tok in moisture_tokens) and ("2" in n and "cm" in n):
            return col

    # Pass 3: raw regex for '2\s*cm'
    for col in df.columns:
        if re.search(r"2\s*cm", str(col), re.IGNORECASE):
            return col

    # No match found
    return None


def main():
    # --- Load data ---
    df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    print(f"Loaded shape: {df.shape}")
    print("Columns:")
    for c in df.columns:
        print(f" - {c}")

    # --- Determine time and theta columns ---
    time_col = TARGET_TIME_COLUMN or detect_time_column(df)
    if time_col is None:
        raise RuntimeError(
            "Could not auto-detect a datetime column. Please set TARGET_TIME_COLUMN to the correct column name."
        )

    if TARGET_THETA_COLUMN is not None:
        theta_col = TARGET_THETA_COLUMN
        if theta_col not in df.columns:
            raise KeyError(f"TARGET_THETA_COLUMN='{theta_col}' not found in columns.")
    else:
        theta_col = detect_2cm_theta_column(df)

    if theta_col is None:
        raise RuntimeError(
            "Could not auto-detect the 2 cm theta/soil moisture column. "
            "Please set TARGET_THETA_COLUMN to the exact column name from the printed columns."
        )

    print(f"\nUsing time column: {time_col}")
    print(f"Using 2 cm theta column: {theta_col}")

    # --- Prepare time index ---
    df["__time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True, infer_datetime_format=True)
    df = df.dropna(subset=["__time"]).sort_values("__time")
    df = df.set_index("__time")

    # --- Plot ---
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, pd.to_numeric(df[theta_col], errors="coerce"), color="tab:blue", linewidth=1.0)
    plt.title("2 cm Theta Time Series")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Theta (2 cm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Ensure output directory exists and save
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=200)
    print(f"Saved figure to: {FIG_PATH}")

    # Also show if running interactively
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
