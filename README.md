# F1 Pit Stop Strategy Predictor

Steven Velasquez — Panther ID 6376609 — CAI4105

Predicts, on a per-lap basis, whether a Formula 1 driver should pit ("Pit")
or continue on track ("Stay Out"), using lap, tire, position, and weather
data pulled from the FastF1 API.

## Requirements

Python 3.9+, with the following libraries:

```
pip install fastf1 pandas numpy scikit-learn matplotlib
```

## Folder contents

```
code/
    main.py               -- single entry point; runs the full pipeline
                              (see "How to run" below)
    pull_f1_data.py         -- pulls raw lap/weather data from FastF1, builds
                              features, labels pit vs. stay-out
    preprocess.py            -- imputes missing values, one-hot encodes tire
                              compound, normalizes numeric features, and
                              splits into train/test HELD OUT BY RACE
    train_evaluate.py        -- trains and compares logistic regression,
                              decision tree, KNN, and SVM; runs 5-fold
                              cross-validation grouped by race; evaluates on
                              the held-out test races

data/
    f1_pitstop_dataset.csv  -- raw pulled dataset, 16,599 laps across 17
                                races spanning the 2022-2025 seasons
    train.csv                -- preprocessed training split (13 races)
    test.csv                  -- preprocessed test split (4 races, held out
                                entirely -- the models never see these races
                                during training or cross-validation)

results/
    model_results.csv         -- accuracy/precision/recall/F1 per model on
                                the held-out test races
    cv_results.csv             -- 5-fold cross-validation F1 (mean/std) per
                                model, computed on the training races only
    preprocessing_summary.txt  -- train/test race split details, imputation
                                notes
    metrics_comparison.png     -- bar chart comparing all four models
    confusion_matrices.png     -- confusion matrix for each model
    predictions.csv            -- per-lap "Pit" / "Stay Out" call from each
                                model on the held-out test races, next to
                                what the driver actually did (Actual) -- the
                                human-readable view of what the models are
                                actually predicting, lap by lap
```

## How to run

The pulled dataset is already included in `data/f1_pitstop_dataset.csv`, so
a single command reproduces every result and plot in the report:

```
cd code
python main.py
```

This reuses the existing dataset and runs preprocessing + training only.
To re-pull fresh data from FastF1 instead (requires internet access, takes
several minutes):

```
python main.py --pull
```

Each stage can also be run individually if needed (`python pull_f1_data.py`,
`python preprocess.py`, `python train_evaluate.py`, in that order, from the
`code/` folder) -- `main.py` is a convenience wrapper around the same three
scripts.

## Data source and scope

Data comes from the FastF1 Python library (official F1 timing feed).
17 races were sampled across the 2022, 2023, 2024, and 2025 seasons
(3-8 races per season) rather than an exhaustive pull of every race, to
keep the pull tractable — the original proposal estimated ~60,000 laps
across full seasons; this project uses 16,599 real laps. See the Final
Report's Discussion section for more on this trade-off and its effect on
cross-validation stability.

## Methodology notes

- Train/test split is done **by race, not by lap** (`GroupShuffleSplit` /
  `GroupKFold` grouped on Season+Round): the four test races are entirely
  unseen during training, which tests the model's ability to generalize to
  a *future* race rather than just unseen laps within a race it has already
  partially learned.
- Class imbalance ("pit" laps are ~3% of all laps): logistic regression,
  decision tree, and SVM use scikit-learn's `class_weight="balanced"`.
  KNN has no such parameter, so its training data is balanced instead via
  random oversampling of the minority class.
- Numeric features are normalized with `StandardScaler` (fit on the
  training races only); tire compound is one-hot encoded; missing values
  (tire compound/age on some wet-affected laps, gap-to-car-ahead for race
  leaders, lap-time degradation on the first lap of a stint) are imputed
  with the median (numeric) or mode (categorical).
