"""
Trains and evaluates the four classifiers (logistic regression, decision
tree, KNN, SVM) on the F1 pit-stop dataset.

Uses:
  - class weighting to handle the "pit" vs "stay out" imbalance
  - 5-fold cross-validation grouped by race (so no fold ever splits a
    single race across train/validation) as an internal check
  - final evaluation on the held-out test races from preprocess.py,
    which the models never see during training or cross-validation

Outputs:
    ../results/model_results.csv
    ../results/cv_results.csv
    ../results/metrics_comparison.png
    ../results/confusion_matrices.png
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.utils import resample

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42


def oversample_minority(X, y):
    """KNN has no built-in class_weight parameter, so we balance its
    training data directly by random oversampling of the minority
    ('pit') class instead."""
    X = np.asarray(X)
    y = np.asarray(y)
    idx_majority = np.where(y == 0)[0]
    idx_minority = np.where(y == 1)[0]
    idx_minority_upsampled = resample(
        idx_minority, replace=True, n_samples=len(idx_majority), random_state=RANDOM_STATE
    )
    idx_combined = np.concatenate([idx_majority, idx_minority_upsampled])
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(idx_combined)
    return X[idx_combined], y[idx_combined]


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=8
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),  # imbalance handled via oversampling, not class_weight
        "SVM": SVC(class_weight="balanced", kernel="rbf", random_state=RANDOM_STATE),
    }


def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    feature_cols = [c for c in train_df.columns
                    if c not in ("Season", "Round", "RaceID", "Driver", "LapNumber", "Pit")]

    X_train = train_df[feature_cols].values
    y_train = train_df["Pit"].values
    groups_train = train_df["RaceID"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["Pit"].values

    MODELS = get_models()

    # -----------------------------------------------------------------
    # 5-fold cross-validation, grouped by race, on the training set only.
    # -----------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    cv_rows = []
    for name, model in MODELS.items():
        fold_f1s = []
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            if name == "KNN":
                X_tr, y_tr = oversample_minority(X_tr, y_tr)

            m = model.__class__(**model.get_params())
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            fold_f1s.append(f1_score(y_val, preds, pos_label=1, zero_division=0))

        cv_rows.append({
            "Model": name,
            "CV_F1_Mean": np.mean(fold_f1s),
            "CV_F1_Std": np.std(fold_f1s),
        })
        print(f"{name}: CV F1 = {np.mean(fold_f1s):.3f} (+/- {np.std(fold_f1s):.3f})")

    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)

    # -----------------------------------------------------------------
    # Fit on full training set, evaluate on held-out test races.
    # -----------------------------------------------------------------
    results = []
    confusion_mats = {}
    all_preds = {}

    for name, model in MODELS.items():
        X_fit, y_fit = (X_train, y_train)
        if name == "KNN":
            X_fit, y_fit = oversample_minority(X_train, y_train)

        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        train_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(X_test)
        predict_time_ms_per_lap = (time.perf_counter() - t0) / len(X_test) * 1000
        all_preds[name] = preds

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, preds, pos_label=1, zero_division=0),
            "F1": f1_score(y_test, preds, pos_label=1, zero_division=0),
            "TrainTimeSeconds": train_time_s,
            "PredictTimeMsPerLap": predict_time_ms_per_lap,
        })
        confusion_mats[name] = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f"\n{name} (held-out test races):")
        print(f"  Accuracy={results[-1]['Accuracy']:.3f}  Precision={results[-1]['Precision']:.3f}  "
              f"Recall={results[-1]['Recall']:.3f}  F1={results[-1]['F1']:.3f}")
        print(f"  Train time: {train_time_s:.3f}s  |  Predict time: {predict_time_ms_per_lap:.4f} ms/lap")
        print(f"  Confusion matrix [[TN FP][FN TP]]:\n{confusion_mats[name]}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)

    # -----------------------------------------------------------------
    # Per-lap predictions
    # -----------------------------------------------------------------
    def label(v):
        return "Pit" if v == 1 else "Stay Out"

    pred_df = test_df[["Season", "Round", "RaceID", "Driver", "LapNumber"]].copy()
    pred_df["Actual"] = test_df["Pit"].map(label)
    for name, preds in all_preds.items():
        col_name = "Predicted_" + name.replace(" ", "")
        pred_df[col_name] = pd.Series(preds).map(label)
    pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    print(f"\nSaved per-lap predictions to {os.path.join(RESULTS_DIR, 'predictions.csv')}")

    # -----------------------------------------------------------------
    # Plot: bar chart comparing metrics across models
    # -----------------------------------------------------------------
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, row) in enumerate(results_df.set_index("Model").iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics], width, label=name)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison on Held-Out Test Races")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # Plot: confusion matrices, 2x2 grid
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, cm) in zip(axes.ravel(), confusion_mats.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Stay Out", "Pit"])
        ax.set_yticklabels(["Stay Out", "Pit"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()