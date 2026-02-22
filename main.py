"""
main.py – Run X-ANFIS on a regression dataset
==================================================
This script trains X-ANFIS, saves the trained model,
and prints a full evaluation report.

To run:
    python main.py
"""

import os
import pickle
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from xanfis import XANFIS


def main():

    # ------------------------------------------------------------------
    # LOAD DATA
    # You can replace this synthetic dataset with your own dataset.
    # Use a synthetic regression dataset.
    # X should be a 2-D array of shape (N, n_features).
    # y should be a 1-D array of shape (N,).
    # ------------------------------------------------------------------
    print("Generating synthetic toy regression dataset...")
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=5,
        noise=0.1,
        random_state=42
    )

    feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    print(f"  Samples  : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}  {feature_names}")

    # ------------------------------------------------------------------
    # PREPROCESS
    # X-ANFIS expects features normalized to [0, 1].
    # ------------------------------------------------------------------
    X = MinMaxScaler().fit_transform(X)
    y = (y - y.min()) / (y.max() - y.min())

    # 70 / 10 / 20 train / val / test split
    X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_val,   y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=42
    )

    print(f"\n  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples\n")

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    model = XANFIS(
        n_rules    = 5,      # number of fuzzy rules
        D_target   = 0.5,     # target pairwise distinguishability 
        lr_perf    = 0.01,    # learning rate — backward (MSE) pass
        lr_xpass   = 0.01,    # learning rate — X-pass (distinguishability)
        lambda_reg = 1e-4,    # L2 regularization on consequent estimation
        patience   = 50,      # early-stopping patience in epochs
    )

    model.fit(X_train, y_train, X_val, y_val, epochs=500)

    # ------------------------------------------------------------------
    # SAVE MODEL (for visualization or later use)
    # ------------------------------------------------------------------

    model_path = "xanfis_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")

    # ------------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------------
    metrics = model.evaluate(X_test, y_test)

    print("\n── Test Results ───────────────────────────────────────────")
    print(f"  {'MSE':<10}: {metrics['MSE']:.4f}")
    print(f"  {'RMSE':<10}: {metrics['RMSE']:.4f}")
    print(f"  {'MAE':<10}: {metrics['MAE']:.4f}")
    print(f"  {'R²':<10}: {metrics['R2']:.4f}")
    print(f"  {'Mean D':<10}: {metrics['mean_D']:.4f}  "
          f"(target = {model.D_target})")
    print("───────────────────────────────────────────────────────────")

    if metrics["mean_D"] >= 0.4:
        print("  ✓ Distinguishability target achieved — fuzzy sets are semantically separable.")
    else:
        print("  ✗ Distinguishability below 0.4 — consider increasing lr_xpass or epochs.")


if __name__ == "__main__":
    main()