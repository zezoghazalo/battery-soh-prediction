"""
data_preprocessing.py
- Loads PulseBat Dataset Excel file (expects /mnt/data/PulseBat Dataset.xlsx)
- Finds feature columns U1..U21 (individual cell voltage measurements)
- Uses the dataset's real SOH column as the prediction target
- Cleans missing values (row-mean imputation; drops rows missing >50% of U values)
- Saves processed CSV to ./results/processed_data.csv
Usage:
    python src/data_preprocessing.py --input "/mnt/data/PulseBat Dataset.xlsx" --output "../results/processed_data.csv"
"""
import argparse
import pandas as pd
import numpy as np
import os

def find_u_columns(df):
    # find columns that start with 'U' followed by 1..21 (case-insensitive)
    candidates = []
    for col in df.columns:
        name = str(col).strip()
        if name.upper().startswith("U"):
            try:
                num = int(''.join(ch for ch in name[1:] if ch.isdigit()))
                if 1 <= num <= 21:
                    candidates.append(col)
            except:
                continue
    def keyfn(c):
        s = ''.join(ch for ch in str(c)[1:] if ch.isdigit())
        return int(s) if s else 0
    return sorted(candidates, key=keyfn)

def main(args):
    inp = args.input
    out = args.output
    if not os.path.exists(inp):
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_excel(inp, sheet_name=0)
    print(f"Loaded sheet with shape {df.shape} and columns: {list(df.columns)[:10]}...")

    ucols = find_u_columns(df)
    if len(ucols) < 1:
        raise RuntimeError("No U1..U21 columns found. Columns detected: " + ", ".join(df.columns.astype(str)))
    print(f"Detected U-columns: {ucols} (count={len(ucols)})")

    # Locate the real SOH target column
    soh_col = next((c for c in df.columns if str(c).strip().upper() == 'SOH'), None)
    if soh_col is None:
        raise RuntimeError("No SOH column found in dataset. Columns: " + ", ".join(df.columns.astype(str)))
    print(f"Using '{soh_col}' as prediction target.")

    # Select feature + target columns and coerce to numeric
    u_df = df[ucols].apply(pd.to_numeric, errors='coerce')
    soh_series = pd.to_numeric(df[soh_col], errors='coerce')

    # Drop rows where target is missing
    valid_mask = soh_series.notna()
    u_df = u_df[valid_mask].copy()
    soh_series = soh_series[valid_mask].copy()

    # Impute missing feature values: drop rows missing >50%, fill remainder with row mean
    row_missing = u_df.isna().sum(axis=1)
    drop_mask = row_missing > (len(ucols) // 2)
    if drop_mask.any():
        print(f"Dropping {drop_mask.sum()} rows with >{len(ucols)//2} missing U values")
    u_df = u_df[~drop_mask].copy()
    soh_series = soh_series[~drop_mask].copy()
    u_df = u_df.apply(lambda row: row.fillna(row.mean()), axis=1)

    # Assemble processed dataframe: target first, then features
    processed = u_df.copy()
    processed.insert(0, 'SOH', soh_series.values)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    processed.to_csv(out, index=False)
    print(f"Saved processed data to {out}. Shape: {processed.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="/mnt/data/PulseBat Dataset.xlsx", help="Path to PulseBat Dataset.xlsx")
    p.add_argument("--output", type=str, default="../results/processed_data.csv", help="CSV output path (relative to src/)")
    args = p.parse_args()
    main(args)
