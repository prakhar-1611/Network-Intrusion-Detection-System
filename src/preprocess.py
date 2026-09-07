import glob
# os lets us build file paths and check/create folders
import os
# pandas is the main library used to load and manipulate the csv data as tables
import pandas as pd
# numpy is used here mainly for handling infinity values and doing the where check
import numpy as np

# these are the columns that identify a specific device or session instead of describing
# how the traffic actually behaved, keeping them would let the model just memorize which
# ip was the attacker in this one 2017 capture instead of learning real patterns
LEAKY_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "SimillarHTTP",
]

# the name of the column that holds the actual answer we are trying to predict
LABEL_COLUMN = "Label"


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # loops through every column name and strips leading/trailing spaces since
    # cicflowmeter writes inconsistent spacing across the different csv files
    df.columns = [c.strip() for c in df.columns]
    # gives back the same dataframe with the fixed column names
    return df


def load_raw_csvs(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate every CSV in raw_dir (the 8 CIC-IDS-2017 files)."""
    # builds the pattern "data/raw/*.csv" and finds every matching file, then sorts
    # them alphabetically so monday always loads before tuesday and so on
    csv_paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    # if nothing was found, stop immediately with a clear error instead of silently
    # continuing with an empty dataset
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download the dataset CSVs "
            "there first (see README.md)."
        )

    # empty list that will hold each file's table before we combine them
    frames = []
    # goes through every csv file path one at a time
    for path in csv_paths:
        # prints the filename currently being loaded so you can see progress
        print(f"Loading {os.path.basename(path)} ...")
        # actually reads this one csv into a pandas table, low_memory=False avoids
        # mixed type guessing on such a big file, latin1 encoding avoids crashing
        # on the broken characters some of these files contain
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        # cleans this file's column names right after loading it
        df = _clean_column_names(df)
        # adds this cleaned table onto the list of tables
        frames.append(df)

    # stacks all 8 tables into one single table, renumbering rows continuously
    full = pd.concat(frames, ignore_index=True)
    # prints how many total rows we ended up with and from how many files
    print(f"Loaded {len(full):,} rows across {len(csv_paths)} files.")
    # hands the combined table back to whoever called this function
    return full


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, drop leakage/identifier columns, handle inf/NaN."""
    # cleans column names again just to be safe, harmless if already done
    df = _clean_column_names(df)

    # only keeps the leaky column names that actually exist in this particular dataframe
    to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    # removes those identifier columns entirely so the model can't cheat using them
    df = df.drop(columns=to_drop)

    # some flow duration values are zero, which makes bytes/duration turn into infinity,
    # this line finds any positive or negative infinity and turns it into a missing value
    df = df.replace([np.inf, -np.inf], np.nan)

    # counts how many rows we have before dropping anything
    before = len(df)
    # removes every row that has even one missing value left in it after the inf replace
    df = df.dropna()
    # counts how many rows remain after dropping
    after = len(df)
    # prints how many rows were dropped and what percentage that is of the original data
    print(f"Dropped {before - after:,} rows with NaN/inf values "
          f"({(before - after) / before:.2%}).")

    # cleans up the text inside the label column since some files have weird spacing
    # or broken characters in place of a normal dash
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        # forces every value to be treated as plain text first
        .astype(str)
        # removes stray spaces from the start and end of each label
        .str.strip()
        # swaps a corrupted dash character for a normal readable dash
        .str.replace("\x96", "-", regex=False)
        # swaps another broken encoding symbol for a normal dash as well
        .str.replace("�", "-", regex=False)
    )

    # counts how many rows are exact duplicates of an earlier row
    dup_count = df.duplicated().sum()
    # only bothers dropping and printing if duplicates actually exist
    if dup_count:
        # removes the duplicate rows, keeping just the first occurrence of each
        df = df.drop_duplicates()
        # prints how many duplicate rows got removed
        print(f"Dropped {dup_count:,} duplicate rows.")

    # renumbers the row index cleanly from 0 upward since dropping rows left gaps
    return df.reset_index(drop=True)


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a BENIGN/ATTACK binary column alongside the multiclass label."""
    # creates a new column that just says benign or attack, collapsing all the
    # specific attack names into one simple category for an easier binary model
    df["Label_Binary"] = np.where(df[LABEL_COLUMN] == "BENIGN", "BENIGN", "ATTACK")
    # returns the dataframe with the new column attached
    return df


def summarize_classes(df: pd.DataFrame) -> pd.Series:
    # counts how many rows belong to each label value, sorted most common first
    return df[LABEL_COLUMN].value_counts()


def run(raw_dir: str, out_path: str) -> pd.DataFrame:
    # loads and combines all the raw csv files into one table
    df = load_raw_csvs(raw_dir)
    # cleans that combined table, dropping bad columns/rows and duplicates
    df = clean_dataset(df)
    # adds the simplified binary label column on top of the multiclass one
    df = add_binary_label(df)

    # prints a header line before showing the class counts
    print("\nClass distribution:")
    # prints how many rows exist per label so you can see the imbalance
    print(summarize_classes(df))

    # makes sure the output folder actually exists before saving into it
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # saves the final cleaned table as a parquet file, a faster format to reload later
    df.to_parquet(out_path, index=False)
    # prints a final confirmation with row and column counts
    print(f"\nSaved processed dataset to {out_path} ({len(df):,} rows, "
          f"{df.shape[1]} columns).")
    # returns the cleaned table in case this function gets called from other code
    return df


if __name__ == "__main__":
    # only imported here since it's only needed when running this file directly
    import argparse

    # sets up a command line argument parser with a short description
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS-2017 CSVs")
    # defines the --raw_dir flag, defaulting to data/raw if not typed
    parser.add_argument("--raw_dir", default="data/raw",
                         help="Folder containing the 8 downloaded CSV files")
    # defines the --out flag, defaulting to the standard processed file location
    parser.add_argument("--out", default="data/processed/cicids2017_clean.parquet",
                         help="Output path for the cleaned parquet file")
    # actually reads whatever flags were typed on the command line
    args = parser.parse_args()
    # kicks off the whole preprocessing pipeline with those two paths
    run(args.raw_dir, args.out)
