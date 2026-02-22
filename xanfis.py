"""
X-ANFIS: Alternating Bi-Objective Optimization for Explainable Neuro-Fuzzy Systems
This module implements X-ANFIS, a gradient-based neuro-fuzzy inference system that
decouples predictive accuracy and semantic explainability into separate gradient passes,
enabling solutions beyond the convex hull of the weighted-scalarization Pareto front.

Architecture Overview (per training epoch):
  1. Forward Pass  – Cauchy MF computation + regularized LSE for consequents
  2. Backward Pass – Gradient descent on MSE to update antecedent centers & scales
  3. X-Pass        – Gradient descent on distinguishability error (centers only)

Key design choices:
  - Cauchy MFs instead of Gaussian: algebraically decaying tails prevent gradient
    explosion when scales are initialized small (i.e., high distinguishability init).
  - Decoupled objectives: X-pass is applied after each backward pass so explainability
    does not corrupt the performance gradient signal.
  - Scales are frozen in X-pass: prevents trivial coverage collapse.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import skfuzzy as fuzz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cauchy_membership(X, centers, scales):
    """
    Compute normalized Cauchy firing strengths.

    Cauchy MF: mu(x) = 1 / (1 + ((x - c) / gamma)^2)

    The product t-norm is applied across features, then outputs are row-normalized
    to form the design matrix used in least-squares estimation.

    Parameters
    ----------
    X       : (N, n_features)
    centers : (n_rules, n_features)
    scales  : (n_rules, n_features)

    Returns
    -------
    M_norm  : (N, n_rules)  – normalized firing strengths
    """
    # (N, n_rules, n_features)  broadcasting
    diffs = X[:, None, :] - centers[None, :, :]
    sq    = (diffs / scales[None, :, :]) ** 2

    # Product t-norm across features -> (N, n_rules)
    M = 1.0 / (1.0 + sq.sum(axis=2))

    # Row-normalize
    M_sum = M.sum(axis=1, keepdims=True)
    M_sum[M_sum == 0] = 1e-8
    return M / M_sum


def _distinguishability(mu1, mu2, s1, s2):
    """
    Pairwise distinguishability between two adjacent fuzzy sets (Jin et al., 2000).

    D = sqrt((mu1 - mu2)^2 + (sigma1 - sigma2)^2)

    A value of 0 means the sets are identical; larger values imply more separation.
    Recommended target range: [0.4, 0.5].
    """
    return np.sqrt((mu1 - mu2) ** 2 + (s1 - s2) ** 2)


def mean_distinguishability(centers, scales):
    """
    Compute mean pairwise distinguishability across all features and adjacent rule pairs.

    Parameters
    ----------
    centers : (n_rules, n_features)
    scales  : (n_rules, n_features)

    Returns
    -------
    float – mean distinguishability over all adjacent pairs and features.
    """
    n_rules, n_features = centers.shape
    d_vals = []
    for f in range(n_features):
        order = np.argsort(centers[:, f])
        for k in range(n_rules - 1):
            i, j = order[k], order[k + 1]
            d_vals.append(_distinguishability(
                centers[i, f], centers[j, f],
                scales[i, f],  scales[j, f]
            ))
    return float(np.mean(d_vals)) if d_vals else 0.0


# ---------------------------------------------------------------------------
# X-ANFIS
# ---------------------------------------------------------------------------

class XANFIS:
    """
    Explainable Adaptive Neuro-Fuzzy Inference System (X-ANFIS).

    Implements a zero-order Takagi-Sugeno fuzzy system trained with an
    alternating bi-objective scheme:

      - Performance objective  (MSE)           → backward pass
      - Explainability objective (distinguishability) → X-pass

    Parameters
    ----------
    n_rules      : int   – number of fuzzy rules (= number of FCM clusters)
    D_target     : float – target pairwise distinguishability; default 0.5
    lr_perf      : float – learning rate for the backward (performance) pass
    lr_xpass     : float – learning rate for the X-pass (explainability)
    lambda_reg   : float – L2 regularization for consequent LSE
    patience     : int   – early-stopping patience (epochs without val improvement)
    clip         : float – gradient clipping threshold for the backward pass
    min_scale    : float – hard lower bound on scale parameters
    """

    def __init__(
        self,
        n_rules    = 5,
        D_target   = 0.5,
        lr_perf    = 0.1,
        lr_xpass   = 0.1,
        lambda_reg = 1e-4,
        patience   = 15,
        clip       = 1.0,
        min_scale  = 0.01,
    ):
        self.n_rules    = n_rules
        self.D_target   = D_target
        self.lr_perf    = lr_perf
        self.lr_xpass   = lr_xpass
        self.lambda_reg = lambda_reg
        self.patience   = patience
        self.clip       = clip
        self.min_scale  = min_scale

        # Learned parameters (set during fit)
        self.centers    = None   # (n_rules, n_features)
        self.scales     = None   # (n_rules, n_features)
        self.consequents = None  # (n_rules,) – zero-order consequent constants

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_fcm(self, X):
        """
        Initialize centers and scales using Fuzzy C-Means clustering.

        Centers  ← FCM cluster prototypes.
        Scales   ← Weighted standard deviation per cluster and feature,
                   clipped at a minimum of min_scale_factor × feature_range
                   to prevent numerical issues.
        """
        N, n_features = X.shape
        min_scale_factor = 0.01

        # FCM expects shape (n_features, N)
        cntr, u, *_ = fuzz.cluster.cmeans(
            X.T, c=self.n_rules, m=2, error=1e-5, maxiter=1000, seed=42
        )
        self.centers = cntr  # (n_rules, n_features)
        self.centers = np.clip(self.centers, 0.0, 1.0)

        # Membership weights per sample: u has shape (n_rules, N)
        u_weights = u.T  # (N, n_rules)

        # Weighted std per cluster and feature
        stds = np.zeros((self.n_rules, n_features))
        for k in range(self.n_rules):
            w = u_weights[:, k]
            w = w / w.sum()
            for f in range(n_features):
                diffs = X[:, f] - self.centers[k, f]
                var   = np.sum(w * diffs ** 2)
                std   = np.sqrt(var)
                # Robustify: re-estimate after clipping 3-sigma outliers
                clipped = np.clip(diffs, -3 * std, 3 * std)
                stds[k, f] = np.sqrt(np.sum(w * clipped ** 2))

        # Enforce minimum scale
        feature_ranges = np.ptp(X, axis=0)
        min_scales     = min_scale_factor * feature_ranges
        self.scales    = np.maximum(stds, min_scales)

    # ------------------------------------------------------------------
    # Gradient passes
    # ------------------------------------------------------------------

    def _backward_pass(self, X, y, M):
        """
        Update antecedent centers and scales via gradient descent on MSE.

        The Cauchy gradient retains a gamma^{-3} term but is moderated by
        mu^2(x), preventing explosion even at small scale initializations.
        """
        N = X.shape[0]
        y_pred = M @ self.consequents
        err    = (y - y_pred)[:, None, None]   # (N, 1, 1)

        diffs = X[:, None, :] - self.centers[None, :, :]   # (N, n_rules, n_features)
        u     = ((diffs / self.scales[None, :, :]) ** 2).sum(axis=2, keepdims=True)

        # dM/dc: partial of unnormalized Cauchy MF w.r.t. center
        dM_dc = -M[:, :, None] / (1.0 + u) * 2 * diffs / (self.scales[None, :, :] ** 2)
        # dM/ds: partial of unnormalized Cauchy MF w.r.t. scale
        dM_ds =  M[:, :, None] / (1.0 + u) * 2 * (diffs ** 2) / (self.scales[None, :, :] ** 3)

        # Chain rule through consequent output
        dY_dc = dM_dc * self.consequents[None, :, None]
        dY_ds = dM_ds * self.consequents[None, :, None]

        grad_c = -2 / N * (err * dY_dc).sum(axis=0)
        grad_s = -2 / N * (err * dY_ds).sum(axis=0)

        # Gradient clipping for stability
        grad_c = np.clip(grad_c, -self.clip, self.clip)
        grad_s = np.clip(grad_s, -self.clip, self.clip)

        self.centers -= self.lr_perf * grad_c
        self.centers = np.clip(self.centers, 0.0, 1.0)
        self.scales   = np.maximum(self.scales - self.lr_perf * grad_s, self.min_scale)

    def _xpass(self):
        """
        X-pass: steer pairwise distinguishability of adjacent fuzzy sets
        toward D_target using gradient descent on the squared error loss:

            L = 0.5 * (D_ij - D_target)^2

        Only centers are updated; scales are frozen to preserve coverage.
        Adjacent sets are identified by sorted center order per feature.
        """
        for f in range(self.centers.shape[1]):
            order = np.argsort(self.centers[:, f])

            for k in range(len(order) - 1):
                i, j = order[k], order[k + 1]

                mu1, mu2 = self.centers[i, f], self.centers[j, f]
                s1,  s2  = self.scales[i, f],  self.scales[j, f]

                D = _distinguishability(mu1, mu2, s1, s2)
                if D < 1e-8:
                    continue

                # Gradient of L w.r.t. each center
                coeff = (D - self.D_target) / D
                self.centers[i, f] -= self.lr_xpass * coeff * (mu1 - mu2)
                self.centers[j, f] -= self.lr_xpass * coeff * (mu2 - mu1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val, epochs=500, verbose=True):
        """
        Train X-ANFIS with alternating bi-objective optimization.

        Each epoch:
          1. Compute Cauchy MFs  →  normalized firing strengths M
          2. Solve regularized LSE for consequents: w = (M'M + λI)^{-1} M'y
          3. Backward pass: gradient descent on MSE (updates centers + scales)
          4. X-pass: gradient descent on distinguishability error (updates centers only)
          5. Early stopping on validation MSE

        Parameters
        ----------
        X_train, y_train : training data
        X_val,   y_val   : validation data used for early stopping
        epochs           : maximum number of training epochs
        verbose          : print progress every 20 epochs
        """
        self._init_fcm(X_train)

        best_val_mse = np.inf
        wait         = 0

        for ep in range(epochs):
            # ---- Forward pass: compute firing strengths + solve consequents ----
            M = _cauchy_membership(X_train, self.centers, self.scales)
            I = np.eye(self.n_rules)
            self.consequents = np.linalg.solve(
                M.T @ M + self.lambda_reg * I, M.T @ y_train
            )

            # ---- Backward pass: update antecedent parameters on MSE ----------
            self._backward_pass(X_train, y_train, M)

            # ---- X-pass: update centers toward target distinguishability ------
            self._xpass()

            # ---- Validation & early stopping ----------------------------------
            val_mse = mean_squared_error(y_val, self.predict(X_val))

            if verbose and ep % 20 == 0:
                train_mse = mean_squared_error(y_train, M @ self.consequents)
                D_mean    = mean_distinguishability(self.centers, self.scales)
                print(f"Epoch {ep+1:4d} | Train MSE: {train_mse:.5f} "
                      f"| Val MSE: {val_mse:.5f} | Mean D: {D_mean:.3f}")

            if val_mse < best_val_mse - 1e-5:
                best_val_mse = val_mse
                wait         = 0
            else:
                wait += 1
            if wait >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {ep+1} | Best Val MSE: {best_val_mse:.5f}")
                break

        return self

    def predict(self, X):
        """
        Predict outputs for input matrix X.

        Parameters
        ----------
        X : (N, n_features)

        Returns
        -------
        y_pred : (N,)
        """
        M = _cauchy_membership(X, self.centers, self.scales)
        return M @ self.consequents

    def evaluate(self, X, y):
        """
        Return a dict of regression metrics and mean distinguishability.

        Metrics: MSE, RMSE, MAE, R², mean_D
        """
        y_pred = self.predict(X)
        mse    = mean_squared_error(y, y_pred)
        return {
            "MSE"    : mse,
            "RMSE"   : float(np.sqrt(mse)),
            "MAE"    : float(mean_absolute_error(y, y_pred)),
            "R2"     : float(r2_score(y, y_pred)),
            "mean_D" : mean_distinguishability(self.centers, self.scales),
        }