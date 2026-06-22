# 5G User Prediction — Logistic Regression vs LightGBM

A machine learning project for predicting 5G user adoption using logistic regression and LightGBM classifiers on a highly imbalanced dataset (1.32% positive class).

## 📁 Project Structure

```
├── LogisticOrLGBM.ipynb          # Baseline: Logistic Regression vs LightGBM comparison
├── LogisticOrLGBM-Batter.ipynb   # Batter: Advanced feature engineering & hyperparameter tuning
├── LogisticOrLGBM-cc.ipynb       # final Version: Data-leakage-free pipeline with best practices
├── train.csv                     # Training dataset (800,000 samples × 60 columns)
└── README.md                     # This file
```

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Samples | 800,000 |
| Features | 58 (20 categorical `cat_0`–`cat_19` + 38 numerical `num_0`–`num_37`) |
| ID column | `id` |
| Target | `target` (0 = non-5G user, 1 = 5G user) |
| Positive class ratio | **1.32%** (10,600 out of 800,000) |
| Imbalance ratio | ~74.5:1 |
| Missing values | None |
| Memory footprint | ~366 MB |

> 📥 **Download `train.csv`:** The dataset file is too large for GitHub. Download it from [QuarkCloud](https://pan.quark.cn/s/54c8f46ccfb2).

## 🎯 Objective

Predict which users are likely to adopt 5G services based on anonymized categorical and numerical features. This is a **binary classification** problem with **extreme class imbalance**.

    ## 📓 Notebook Summaries

### 1. `LogisticOrLGBM.ipynb` — Baseline

A straightforward comparison between Logistic Regression and LightGBM.

**Pipeline:**
- Missing value imputation (mode for categorical, median for numerical)
- 70/30 stratified train-test split
- StandardScaler for Logistic Regression
- Default and tuned LightGBM

**Models & AUC:**

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.8352 |
| LightGBM (default) | 0.9054 |
| LightGBM (tuned) | 0.9128 |

**Key insight:** LightGBM significantly outperforms Logistic Regression. Tuning `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8` yields the best result in this baseline.

---

### 2. `LogisticOrLGBM-Batter.ipynb` — Batter (Advanced Exploration)

Explores a wide range of strategies to boost performance beyond the baseline.

**Techniques attempted:**

| Strategy | AUC | Verdict |
|----------|-----|--------|
| Class weight balancing (`scale_pos_weight`) | 0.9036 | ❌ Harmful |
| SMOTE oversampling | 0.8280 | ❌ Significantly worse |
| Random undersampling | 0.9086 | ⚠️ Mild improvement but loses data |
| Log / square / sqrt transforms on numerical features | 0.9125 | ⚠️ Marginal (tree models don't benefit) |
| Feature crosses (cat×cat, num×num) | 0.9142 | ✅ Noticeable gain |
| Group statistics (cat×num mean/std) | 0.9149 | ✅ Best single improvement |
| Feature selection (Top-30/40/50/60) | 0.9054–0.9133 | ✅ Compresses model without loss at Top-50 |
| Variance threshold filtering | 0.9123 | ✅ Removes noise |
| RandomizedSearchCV (50 trials, 5-fold) | 0.9137 | ✅ Best hyperparameters found |

**Best hyperparameters from RandomizedSearchCV:**
```python
{
    'subsample': 1.0,
    'n_estimators': 500,
    'min_child_samples': 30,
    'max_depth': 5,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8
}
```

**⚠️ Critical issue identified:** Group statistics and some encoding steps were performed *before* the train-test split, causing **data leakage** and inflating AUC estimates.

---

### 3. `LogisticOrLGBM-cc.ipynb` — final Version (Production-Ready)

The most rigorously engineered version, built on lessons learned from Batter.

**Key improvements:**

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Split-first pipeline** | Train/test split before any feature engineering to prevent data leakage |
| 2 | **Target Encoding (KFold)** | 5-fold out-of-fold target encoding for all 20 categorical features |
| 3 | **Count Encoding** | Category frequency encoding |
| 4 | **Group statistics** | cat×num mean & std (validated as the most effective feature engineering) |
| 5 | **Extended group stats** | Importance-based selection for broader group statistics |
| 6 | **Directional feature crosses** | Only Top-5 features crossed to avoid dimension explosion |
| 7 | **L1/L2 regularization** | `reg_alpha=0.1`, `reg_lambda=1.0` to prevent overfitting |
| 8 | **Variance filtering** | Removed 23 low-variance features (148 → 125) |
| 9 | **Stratified 5-Fold CV** | Robust performance estimation |
| 10 | **Top-50 feature selection** | Compress model by ~60% without performance loss |

**Final Results:**

| Model | AUC | Avg Precision | Precision | Recall | F1 |
|-------|-----|---------------|-----------|--------|-----|
| Baseline LightGBM (58 features) | 0.9128 | 0.1448 | 0.2742 | 0.0053 | 0.0105 |
| **final Version (125 features)** | **0.9166** | **0.1668** | **0.4107** | **0.0072** | **0.0142** |
| Final Top-50 features | ~0.916 | — | — | — | — |

**5-Fold Cross-Validation:**
- Mean AUC: **0.9153**
- Std Dev: **0.0018** (very stable)

## 📈 Performance Progression

```
Logistic Regression (baseline)    0.8352  ████████▌
LightGBM default                  0.9054  ████████████████▌
LightGBM tuned                    0.9128  ███████████████████▌
Batter best (group stats)         0.9149  ████████████████████
final Version                     0.9166  ████████████████████▌
```

## ⚠️ Important Notes on Imbalanced Data

- **AUC can be misleading.** On this dataset (1.32% positive), a model that predicts "0" for every sample still achieves ~99% accuracy and can show deceptively high AUC.
- **Average Precision** and **Recall** are more honest metrics for imbalanced classification.
- The low recall (~0.01) across all models indicates that identifying 5G users is inherently difficult with the given features — this is typical when the signal is weak relative to the class imbalance.

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib imbalanced-learn
```

### Run a Notebook

1. Open any of the `.ipynb` files in Jupyter Notebook / JupyterLab / VS Code.
2. Run all cells in order.
3. For the Final Version, trained models are saved as:
   - `lgb_model_cc.pkl` — Full-feature model
   - `lgb_model_cc_top50.pkl` — Lightweight Top-50 feature model

### Load a Saved Model

```python
import joblib

# Load full-feature model
model = joblib.load('lgb_model_cc.pkl')

# Or load the lightweight version
bundle = joblib.load('lgb_model_cc_top50.pkl')
model = bundle['model']
features = bundle['features']

# Predict on new data
predictions = model.predict_proba(X_new[features])[:, 1]
```

## 🔑 Key Takeaways

1. **LightGBM >> Logistic Regression** on this tabular data with mixed feature types.
2. **Feature engineering matters more than model complexity.** Group statistics (cat×num aggregations) provided the single largest AUC improvement.
3. **Don't trust AUC blindly on imbalanced data.** Always check Average Precision, Precision, Recall, and the confusion matrix.
4. **SMOTE and class weighting can backfire.** On this dataset, both techniques reduced AUC — likely because they introduced noise that outweighed the benefit of balancing.
5. **Split before engineering.** The Batter notebook's most valuable contribution was revealing the data leakage pattern, corrected in the Final Version.
6. **Log/sqrt transforms don't help tree models.** Statistical transformations on numerical features had negligible effect on LightGBM (unlike what they'd do for linear models).
7. **Feature selection compresses models without loss.** Top-50 features achieve comparable AUC to 125 features, enabling faster inference.

## 📄 License

This project is for educational and research purposes.

---

*Notebook content is in Chinese; this README provides an English overview of the methodology, results, and key findings.*
