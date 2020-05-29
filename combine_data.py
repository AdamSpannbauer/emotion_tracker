import glob
import pandas as pd

# Read and combine all relevant csvs
csv_paths = glob.glob("data/*.csv")
full_df = pd.concat(pd.read_csv(p, low_memory=False) for p in csv_paths)
full_df.to_csv("data/combined.csv", index=False)
