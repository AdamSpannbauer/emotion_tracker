import glob
import pandas as pd

# Read and combine all relevant csvs
csv_paths = sorted(glob.glob("data/2*.csv"))
full_df = pd.concat((pd.read_csv(p, low_memory=False) for p in csv_paths), sort=False)
full_df.to_csv("data/combined.csv", index=False)
