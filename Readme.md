# X-ANFIS: Alternating Bi-Objective Optimization for Explainable Neuro-Fuzzy Systems

---

## Overview

Fuzzy systems are naturally interpretable through their rule-based structure and linguistic variables, but standard training methods tend to collapse membership functions into overlapping, indistinguishable sets — sacrificing semantic clarity for predictive performance.

**X-ANFIS** addresses this with an *alternating bi-objective optimization* scheme that keeps the two objectives — accuracy and explainability — decoupled across separate gradient passes within each training epoch:


| Pass              | What it does                                                               | Parameters updated        |
| ----------------- | -------------------------------------------------------------------------- | ------------------------- |
| **Forward pass**  | Compute Cauchy membership functions; solve regularized LSE for consequents | Consequents (closed-form) |
| **Backward pass** | Gradient descent on MSE                                                    | Centers + Scales          |
| **X-pass**        | Gradient descent on pairwise distinguishability error                      | Centers only              |


This decoupling, combined with Cauchy membership functions, allows X-ANFIS to recover solutions that lie **beyond the convex hull** of what weighted-scalarization multi-objective methods can reach.

---

## Key Design Choices

### Cauchy Membership Functions

Standard Gaussian MFs suffer from gradient explosion when scales are initialized small:

$$\left|\frac{\partial \mathcal{L}}{\partial \sigma}\right| \propto \sigma^{-3} \to \infty \quad \text{as } \sigma \to 0$$

Small-scale initialization is desirable because it promotes high distinguishability from the start. The Cauchy MF:

$$\mu(x) = \frac{1}{1 + \left(\frac{x - c}{\gamma}\right)^2}$$

has algebraically decaying tails and a moderating $\mu^2(x)$ factor in its gradient that prevents explosion even at narrow initializations.

### Alternating Gradient Passes (not scalarization)

Weighted scalarization ($\mathcal{L} = \mathcal{L}*{perf} + \alpha \mathcal{L}*{expl}$) cannot recover non-convex Pareto regions. X-ANFIS instead applies the two gradient signals sequentially and independently, each with its own learning rate, avoiding the objective-scale interference and aggregation artifacts of scalarization.

### X-Pass: Pairwise Distinguishability

For each feature, adjacent fuzzy sets (ordered by center) are pushed toward a target pairwise distinguishability:

$$D_{ij} = \sqrt{(\mu_i - \mu_j)^2 + (\sigma_i - \sigma_j)^2}$$

$$\mathcal{L}*{X} = \tfrac{1}{2}(D*{ij} - D_{\text{target}})^2$$

Scales are **frozen** during the X-pass to prevent coverage collapse (i.e., sets shrinking to avoid overlap instead of spacing out).

---

## Repo Structure

```
XANFIS/
│
├── xanfis.py          # Architecture — XANFIS class, Cauchy MFs, helpers
├── main.py            # Entry point — load data, train, evaluate, save model
├── visualizer.py      # Plot learned Cauchy membership functions per feature
├── requirements.txt
└── README.md
```


| File            | Purpose                                                         |
| --------------- | --------------------------------------------------------------- |
| `xanfis.py`     | The architecture. Read this to understand the model.            |
| `main.py`       | Run this to train and evaluate on your dataset.                 |
| `visualizer.py` | Run this after training to inspect the learned fuzzy partition. |


---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scikit-learn`, `scikit-fuzzy`, `matplotlib`

---

## Usage

### Train

```bash
python main.py
```

Trains X-ANFIS on a synthetic regression dataset, prints evaluation metrics, and saves the model to `xanfis_model.pkl`. To use your own data, replace the `LOAD DATA` block in `main.py` with your own `X` / `y` arrays — everything else stays the same.

### Visualize

```bash
python visualizer.py                      # loads xanfis_model.pkl by default
python visualizer.py path/to/model.pkl    # or point to a specific file
```

Produces one plot per feature showing the learned Cauchy membership functions for all rules, making the fuzzy partition directly inspectable.

### API

```python
from xanfis import XANFIS

model = XANFIS(
    n_rules    = 5,      # number of fuzzy rules (keep low for interpretability)
    D_target   = 0.5,   # target pairwise distinguishability  
    lr_perf    = 0.01,   # learning rate for backward (MSE) pass
    lr_xpass   = 0.01,   # learning rate for X-pass
    lambda_reg = 1e-4,  # L2 regularization on consequents
    patience   = 50,    # early-stopping patience (epochs)
)

# Features must be normalized to [0, 1]  (MinMaxScaler recommended)
model.fit(X_train, y_train, X_val, y_val, epochs=500)

metrics = model.evaluate(X_test, y_test)
# returns: MSE, RMSE, MAE, R², mean_D (mean pairwise distinguishability)
```

---

## Hyperparameter Notes


| Parameter  | Typical range | Effect                                                                                |
| ---------- | ------------- | ------------------------------------------------------------------------------------- |
| `n_rules`  | 5–10          | Fewer rules → more interpretable; more rules → higher capacity                        |
| `D_target` | 0.1–0.5       | High → more separated fuzzy sets. higher rule count demands lower distinguishability. |
| `lr_perf`  | 0.01–0.1      | Learning rate for the MSE backward pass                                               |
| `lr_xpass` | 0.01–0.1      | Higher values converge faster but may oscillate                                       |
| `patience` | 10–50         | Early stopping is based on validation MSE, not distinguishability                     |


> **Note on rule count vs. distinguishability:** enforcing high distinguishability across many rules in a bounded input space creates tension between coverage and semantic coherence. Start with 5 rules and scale up only if predictive capacity is insufficient. 

---

## Explainability Metrics

`mean_distinguishability(centers, scales)` computes the mean $D_{ij}$ across all adjacent fuzzy set pairs and features. The visualizer provides a qualitative complement — well-separated, non-overlapping curves per feature are a direct visual indicator of a semantically coherent partition.

---

## Citation

**If you use X-ANFIS in your research or find this repository useful, please cite:**

Khaled, Q., Kaymak, U., & Genga, L. (2026).
Alternating Bi-Objective Optimization for Explainable Neuro-Fuzzy Systems.
arXiv:2602.19253 [cs.LG].
[https://doi.org/10.48550/arXiv.2602.19253](https://doi.org/10.48550/arXiv.2602.19253)



---

