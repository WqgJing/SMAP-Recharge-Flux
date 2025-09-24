import os
import pandas as pd
from pathlib import Path
import requests, re
from bs4 import BeautifulSoup
import re


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

    print(f"Found {len(df)} CSV files.")
    return df


onedrive_folder = "~/OneDrive - Georgia Institute of Technology/USGS_GW_data"
df_files = list_csv_files(onedrive_folder)


pattern = re.compile(r"^well_(\d+)_")

# Extract site_no column
df_files["site_no"] = df_files["filename"].apply(
    lambda x: pattern.search(x).group(1) if pattern.search(x) else None
)

# Save all site numbers to a Python variable (list)
site_nos = df_files["site_no"].dropna().unique().tolist()


def get_latlon_from_inventory(site_no):
    url = f"https://waterdata.usgs.gov/nwis/inventory/?site_no={site_no}&agency_cd=USGS"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)

    match = re.search(
        r"Latitude\s+([\d°\'\.]+\"?),\s*Longitude\s+([\d°\'\.]+\"?)", text
    )
    if match:
        return match.group(1), match.group(2)
    return None, None


# Loop through all site numbers
results = []
for s in site_nos[:10]:  # <-- site_nos is your list of extracted site numbers
    lat, lon = get_latlon_from_inventory(s)
    results.append({"site_no": s, "latitude_dms": lat, "longitude_dms": lon})

# Save to a variable (DataFrame in memory)
df_coords = pd.DataFrame(results)

# Also save to a plain list if you prefer
coords_list = results

# Save coords_list to CSV
coords_df = pd.DataFrame(coords_list)
coords_df.to_csv(
    "/Users/jjason3/Downloads/Recharge project/Data/well_coordinates.csv", index=False
)

print(df_coords.head())
