# Credit Card Fraud Detection via Model Retraining & Fine‑Tuning

![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)

A research‑grade exploration of how **temporal validation** and **continual learning** impact credit‑card fraud detection performance.  
Built for CSE6363 (Machine Learning) — University of Texas at Arlington.

---

## ✨ Project Highlights
- **Dataset:** [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284 807 rows, 492 fraud ≈ 0.172 %)
- **Models:** Logistic Regression, XGBoost, PyTorch MLP
- **Experiments:**  
  1. Baseline random 80 / 20 split  
  2. Temporal retraining (train early → test late)  
  3. Fine‑tuning (60 / 20 / 20 rolling window)
- **Key Metric:** PR‑AUC (plus ROC‑AUC, precision & recall)
- **Key Insight:** XGBoost excels in precision; MLP benefits most from fine‑tuning.

---

## 🚀 Quick Start
```bash
# 1. Clone
git clone https://github.com/your‑username/credit‑card‑fraud‑detection.git
cd credit‑card‑fraud‑detection

# 2. Create Conda env
conda env create -f environment.yml
conda activate fraud-detect-env

# 3. Launch notebook
jupyter lab notebook.ipynb
```

> 🔑 **Tip:** Place `creditcard.csv` in `./data/` or update the path in the notebook.

---

## 🔬 Results Snapshot

| Model | PR‑AUC | ROC‑AUC | Precision | Recall |
|-------|--------|---------|-----------|--------|
| Logistic Regression | 0.7067 | 0.9713 | 0.0653 | 0.8776 |
| **XGBoost** | **0.8322** | **0.9838** | **0.8864** | 0.7959 |
| MLP (Neural Net) | 0.6911 | 0.9748 | 0.0724 | **0.8980** |

*(Baseline 80/20 split — see full report for temporal & fine‑tuned metrics.)*

---

## 📊 Visuals
Baseline and temporal comparison tables are available in `/reports/`.

---

## 📝 License
This project is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
Feel free to share and adapt with proper attribution.

---

## 🙏 Acknowledgements
- **Dataset:** ULB Machine Learning Group & Worldline  
- **Libraries:** scikit‑learn, XGBoost, PyTorch, seaborn, matplotlib  

