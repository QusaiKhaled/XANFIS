"""
visualize_membership_functions.py
=================================
Visualize learned Cauchy membership functions of a trained X-ANFIS model.

Default behavior:
    Automatically loads: xanfis_model.pkl

Optional:
    python visualize_membership_functions.py path/to/model.pkl
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_MODEL_PATH = "xanfis_model.pkl"


def plot_membership_functions(model, feature_names=None, num_points=400):
    """
    Plot Cauchy membership functions for each feature.

    Parameters
    ----------
    model : trained XANFIS / ANFIS object
        Must expose:
            model.centers  -> (n_rules, n_features)
            model.scales   -> (n_rules, n_features)

    feature_names : list[str] or None
        Names of features. If None, generic names are used.

    num_points : int
        Resolution of curves.
    """

    n_rules, n_features = model.centers.shape

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

    print(f"\nVisualizing {n_rules} rules across {n_features} features")

    for f_idx in range(n_features):

        plt.figure(figsize=(8, 5))

        # Plot range: cover all fuzzy sets
        f_min = np.min(model.centers[:, f_idx] - 3 * model.scales[:, f_idx])
        f_max = np.max(model.centers[:, f_idx] + 3 * model.scales[:, f_idx])

        x_vals = np.linspace(f_min, f_max, num_points)

        for r_idx in range(n_rules):

            c = model.centers[r_idx, f_idx]
            s = model.scales[r_idx, f_idx]

            # Cauchy membership function
            mu = 1.0 / (1.0 + ((x_vals - c) / s) ** 2)

            plt.plot(x_vals, mu, label=f"Rule {r_idx + 1}")

        plt.title(f"Membership Functions — {feature_names[f_idx]}")
        plt.xlabel(feature_names[f_idx])
        plt.ylabel("Membership Degree")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()


def load_model(path):
    """Load a pickled model."""
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {path}")
        print("Ensure xanfis_model.pkl is in the current directory")
        sys.exit(1)


def main():

    # Use provided path if given, otherwise default
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # OPTIONAL: Replace with real feature names
    feature_names = None

    plot_membership_functions(model, feature_names)


if __name__ == "__main__":
    main()