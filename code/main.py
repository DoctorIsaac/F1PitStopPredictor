"""
Single entry point for the F1 pit-stop prediction pipeline.

Results land in ../results/ (relative to this file): model_results.csv,
cv_results.csv, preprocessing_summary.txt, metrics_comparison.png,
confusion_matrices.png.
"""

import argparse

import pull_f1_data
import preprocess
import train_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Run the F1 pit-stop prediction pipeline end to end."
    )
    parser.add_argument(
        "--pull", action="store_true",
        help="Re-pull raw data from FastF1 before preprocessing (requires "
             "internet, slow). If omitted and the dataset file already "
             "exists, the existing file is reused."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1/3: Data")
    print("=" * 70)
    import os
    if args.pull or not os.path.exists(pull_f1_data.OUTPUT_FILE):
        pull_f1_data.main()
    else:
        print(f"Using existing dataset at {pull_f1_data.OUTPUT_FILE} "
              f"(pass --pull to re-fetch from FastF1 instead).")

    print("\n" + "=" * 70)
    print("STEP 2/3: Preprocessing")
    print("=" * 70)
    preprocess.main()

    print("\n" + "=" * 70)
    print("STEP 3/3: Training and evaluating models")
    print("=" * 70)
    train_evaluate.main()

    print("\nPipeline complete. See ../results/ for metrics and plots.")


if __name__ == "__main__":
    main()
