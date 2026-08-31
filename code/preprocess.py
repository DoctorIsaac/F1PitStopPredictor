"""
Preprocessing pipeline for the F1 pit-stop dataset.


Reads ../data/f1_pitstop_dataset.csv and creates cleaned,
encoded, and scaled train/test splits using race-level holdout
to evaluate on unseen races.

Output files:
    ../data/train.csv
    ../data/test.csv
    ../results/preprocessing_summary.txt
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_FILE = os.path.join(DATA_DIR, "f1_pitstop_dataset.csv")
RANDOM_STATE = 42
TEST_RACE_FRACTION = 0.20  # ~20% of races held out entirely for testing


def main():
    df = pd.read_csv(RAW_FILE)
    n_races_total = df.groupby(["Season", "Round"]).ngroups

    df["RaceID"] = df["Season"].astype(str) + "_" + df["Round"].astype(str)

    # --- Imputation --------------------------------------------------------
    compound_mode = df["Compound"].mode()[0]
    df["Compound"] = df["Compound"].fillna(compound_mode)

    for col in ["TyreLife", "GapToCarAhead", "LapTimeDegradation"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # --- Encoding ------------------------------------------------------
    df["Rainfall"] = df["Rainfall"].astype(int)
    compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound")
    df = pd.concat([df, compound_dummies], axis=1)

    feature_cols = (
        ["TyreLife", "Position", "GapToCarAhead", "LapTimeSeconds",
         "LapTimeDegradation", "AirTemp", "TrackTemp", "Humidity",
         "Rainfall", "WindSpeed"]
        + list(compound_dummies.columns)
    )

    # --- Race-level train/test split ------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_RACE_FRACTION, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["RaceID"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_races = sorted(train_df["RaceID"].unique())
    test_races = sorted(test_df["RaceID"].unique())

    # --- Normalization (fit on train only, applied to both) ---------------
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # --- Save --------------------------------------------------------------
    keep_cols = ["Season", "Round", "RaceID", "Driver", "LapNumber"] + feature_cols + ["Pit"]
    train_df[keep_cols].to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_df[keep_cols].to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    summary_lines = [
        f"Total races: {n_races_total}",
        f"Train races ({len(train_races)}): {train_races}",
        f"Test races ({len(test_races)}, held out entirely): {test_races}",
        f"Train rows: {len(train_df)}  (pit rate {train_df['Pit'].mean():.3%})",
        f"Test rows: {len(test_df)}  (pit rate {test_df['Pit'].mean():.3%})",
        f"Compound imputed with mode: {compound_mode}",
        f"Feature columns ({len(feature_cols)}): {feature_cols}",
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    with open(os.path.join(RESULTS_DIR, "preprocessing_summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
