import os
import pandas as pd
from pathlib import Path


def list_csv_files(root_folder, out_csv="csv_file_list.csv"):
    """
    Recursively find all .csv files under root_folder.
    Save results to a CSV and return as a DataFrame.
    """
    root = Path(root_folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    # Collect all .csv files recursively
    file_paths = [str(p) for p in root.rglob("*.csv")]

    # Make a DataFrame
    df = pd.DataFrame({"file_path": file_paths})
    df["filename"] = df["file_path"].apply(lambda x: os.path.basename(x))
    df.to_csv(out_csv, index=False)

    print(f"Found {len(df)} CSV files. Saved list to {out_csv}")
    return df


# Example usage:
# Replace with your actual OneDrive folder path
onedrive_folder = "~/OneDrive - Georgia Institute of Technology/USGS_GW_data"
df_files = list_csv_files(onedrive_folder)

import re

pattern = re.compile(r"^well_(\d+)_")

# Extract site_no column
df_files["site_no"] = df_files["filename"].apply(
    lambda x: pattern.search(x).group(1) if pattern.search(x) else None
)

# Save all site numbers to a Python variable (list)
site_nos = df_files["site_no"].dropna().unique().tolist()

print(len(site_nos), "unique site numbers extracted")
print(site_nos[:10])  # preview first 10
urls = [
    f"https://waterdata.usgs.gov/nwis/inventory/?site_no={s}&agency_cd=USGS"
    for s in site_nos
]
print(urls[:5])  # preview first 5
