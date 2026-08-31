"""
Pulls lap-level Formula 1 data via the FastF1 API and builds a single CSV
ready for the pit-stop classification project.

Output:
    ../data/f1_pitstop_dataset.csv
"""

import os
import fastf1
import pandas as pd
import numpy as np


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(CODE_DIR, "ff1_cache")
OUTPUT_FILE = os.path.join(DATA_DIR, "f1_pitstop_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configure which races to pull. (year, round_number)
# ---------------------------------------------------------------------------
RACES = [
    # 2023
    (2023, 1),   # Bahrain
    (2023, 4),   # Azerbaijan (street circuit)
    (2023, 7),   # Monaco (street circuit, low overtaking)
    (2023, 10),  # Great Britain (Silverstone)
    (2023, 13),  # Monza (high speed)
    (2023, 16),  # Singapore (street, night race)
    (2023, 19),  # Japan (Suzuka)
    (2023, 22),  # Brazil (Interlagos)

    # 2022
    (2022, 1),   # Bahrain (season opener)
    (2022, 11),  # Austria (mid-season)
    (2022, 18),  # Japan (Suzuka, late season)

    # 2024
    (2024, 1),   # Bahrain (season opener)
    (2024, 8),   # Monaco (street circuit)
    (2024, 19),  # United States (Austin, late season)

    # 2025
    (2025, 1),   # season opener
    (2025, 11),  # mid-season
    (2025, 21),  # late season
]

fastf1.Cache.enable_cache(CACHE_DIR)


def build_race_dataframe(year: int, round_number: int) -> pd.DataFrame:
    print(f"Loading {year} round {round_number} ...")
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps.copy()
    weather = session.weather_data.copy()

    if laps.empty:
        print(f"  no lap data for {year} round {round_number}, skipping")
        return pd.DataFrame()

    # --- Basic cleaning -----------------------------------------------
    laps = laps[laps["LapTime"].notna()].copy()

    # Drop laps affected by Safety Car (4) or Red Flag (5) track status,
    # following the same filtering approach used in Hettmann (2024).
    laps = laps[~laps["TrackStatus"].fillna("").str.contains("4|5")]

    # Drop the very first lap of the race (standing start, not a strategic
    # pit decision point).
    laps = laps[laps["LapNumber"] > 1]

    # Drop "out laps" (the lap right after leaving the pits) -- the pit
    # decision was already made on the previous lap.
    laps = laps[laps["PitOutTime"].isna()]

    if laps.empty:
        return pd.DataFrame()

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # --- Target label ----------------------------------------------------
    laps["Pit"] = laps["PitInTime"].notna().astype(int)

    # --- Gap to car ahead --------------------------------------------------
    laps = laps.sort_values(["LapNumber", "Position"])
    laps["GapToCarAhead"] = (
        laps.groupby("LapNumber")["Time"].diff().dt.total_seconds()
    )

    # --- Lap time degradation ----------------------------------------------
    laps = laps.sort_values(["Driver", "Stint", "LapNumber"])
    laps["RollingAvg"] = (
        laps.groupby(["Driver", "Stint"])["LapTimeSeconds"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
    )
    laps["LapTimeDegradation"] = laps["LapTimeSeconds"] - laps["RollingAvg"]

    # --- Weather merge -------------------------------------------------
    laps = laps.sort_values("Time")
    weather = weather.sort_values("Time")
    laps = pd.merge_asof(
        laps,
        weather[["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]],
        on="Time",
        direction="nearest",
    )

    out = laps[[
        "Driver", "LapNumber", "Compound", "TyreLife", "Position",
        "GapToCarAhead", "LapTimeSeconds", "LapTimeDegradation",
        "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed",
        "Pit",
    ]].copy()

    out.insert(0, "Round", round_number)
    out.insert(0, "Season", year)

    print(f"  {len(out)} usable laps")
    return out


def main():
    all_frames = []
    for year, rnd in RACES:
        try:
            df = build_race_dataframe(year, rnd)
            if not df.empty:
                all_frames.append(df)
        except Exception as e:
            print(f"  FAILED on {year} round {rnd}: {e}")

    if not all_frames:
        print("No data collected. Check your internet connection / FastF1 install.")
        return

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Total rows: {len(full)}")
    print(f"Pit rate: {full['Pit'].mean():.3%}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
