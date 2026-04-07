# Quantum Machine Learning for Wildfire Risk Prediction

> **Deloitte Quantum Sustainability Challenge 2026 — Task 1**

A hybrid quantum-classical machine learning pipeline that predicts wildfire occurrence across California zip codes using historical fire incident and weather data. Built with IBM Qiskit and scikit-learn.

## Problem Statement

Predict whether a wildfire will occur in each California zip code in **2023**, based on historical fire and weather data from **2018–2022**. The model uses a Variational Quantum Classifier (VQC), Quantum Kernel SVM, and a novel Hybrid Quantum-Classical model, benchmarked against classical baselines.

## Project Structure

```
quantum-wildfire-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                          # Original competition datasets
│   │   ├── wildfire_weather.csv
│   │   └── Feature_Description_FireHistory_Census.csv
│   └── processed/                    # Engineered features (auto-generated)
│       └── wildfire_features.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data loading & feature engineering
│   ├── classical_models.py           # Classical baseline training & evaluation
│   ├── quantum_models.py             # VQC, Quantum Kernel & Hybrid model
│   └── evaluation.py                 # Full pipeline, comparison & visualization
├── results/                          # Model outputs, plots & predictions
│   ├── model_comparison.png
│   ├── wildfire_predictions_2023.csv
│   ├── classical_results.json
│   ├── quantum_results.json
│   └── comparison_summary.json
└── docs/
    └── approach.md                   # Detailed methodology write-up
```

## Approach

### Pipeline

1. **Data Preprocessing** — Merge fire incidents with monthly weather observations. Engineer 27 features per zip code per year (20 weather + 7 fire history).
2. **Dimensionality Reduction** — PCA from 27 → 8 features (92.5% variance retained) to match qubit count.
3. **Classical Baselines** — Logistic Regression, Random Forest, Gradient Boosting on full 27 features.
4. **Quantum Models** — VQC and Quantum Kernel SVM on 8 PCA features using Qiskit's StatevectorSampler.
5. **Hybrid Model** — Quantum kernel computes 5 similarity features (e.g., similarity to known fire zones), combined with 27 classical features and fed into Gradient Boosting with **threshold tuning** for optimal fire recall.
6. **Evaluation** — F1, AUC-ROC, accuracy, confusion matrices. Full classical vs. quantum comparison.

### Feature Engineering (27 features)

**Weather features (20):** Annual and seasonal temperature/precipitation stats including summer fire-season metrics (Jun–Sep), fall season (Sep–Nov), aridity indices, and dry month counts.

**Fire history features (7):** Cumulative fire count, total/max/avg acres burned, years with fires, fire recency (`had_fire_last_year`), and fire frequency.

### Quantum Circuit

| Component | Configuration |
|-----------|--------------|
| **Qubits** | 8 |
| **Feature Map** | ZZFeatureMap (linear entanglement, encodes pairwise feature interactions) |
| **Ansatz (VQC)** | RealAmplitudes (2 reps, linear entanglement, 24 trainable parameters) |
| **Kernel (QSVM)** | Fidelity-based quantum kernel with ZZFeatureMap (2 reps) |
| **Hybrid Model** | Quantum kernel similarity features + classical features → Gradient Boosting |
| **Simulator** | StatevectorSampler (exact simulation) |

### Results Summary

| Model | Type | Features | Qubits | Test F1 | Test AUC | Accuracy | Fire Recall |
|-------|------|----------|--------|---------|----------|----------|-------------|
| Gradient Boosting | Classical | 27 | — | 0.425 | 0.880 | 93% | 50% |
| **Hybrid (Classical + Quantum)** | **Hybrid** | **27 + 5** | **8** | **0.348** | **0.862** | **88%** | **72%** |
| Quantum Kernel SVM | Quantum | 8 PCA | 8 | 0.130 | — | 41% | 94% |
| VQC | Quantum | 8 PCA | 8 | 0.103 | — | 57% | 51% |

The hybrid model uses **threshold tuning on the validation set** to optimize for fire recall — prioritizing safety (catching real fires) over precision (avoiding false alarms). For wildfire prediction, missing a fire is far more costly than checking on a safe zip code.

### Key Result: Catching 72% of Real Fires

Out of 118 actual wildfire zip codes in 2023, the hybrid model correctly identified **85 (72%)** while maintaining 88% overall accuracy. This is a major improvement over the default 0.5 threshold which only caught 36% of fires.

| Metric | Default Threshold | Tuned Threshold |
|---|---|---|
| Fires correctly caught | 43/118 (36%) | **85/118 (72%)** |
| False alarms | 63 | 286 |
| Test F1 | 0.405 | 0.348 |
| Test AUC | 0.885 | 0.862 |
| Accuracy | 93% | 88% |

### Quantum Features Add Real Value

In the hybrid model, all 5 quantum-derived features ranked in the **top 15 most important features**, with `q_max_sim_fire` (maximum quantum kernel similarity to known fire zones) ranking **2nd overall** at 17.5% importance — higher than most classical features.

| Rank | Feature | Importance | Source |
|------|---------|------------|--------|
| 1 | hist_avg_acres | 21.0% | Classical |
| **2** | **q_max_sim_fire** | **17.5%** | **Quantum** |
| 3 | hist_max_acres | 10.1% | Classical |
| 4 | hist_total_acres | 9.2% | Classical |
| **5** | **q_sim_ratio** | **5.4%** | **Quantum** |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/narendrarane50/Quantum-Sustainability-Challenge.git
cd Quantum-Sustainability-Challenge
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add data

Place the competition datasets in `data/raw/`:
- `wildfire_weather.csv`
- `Feature_Description_FireHistory_Census.csv`

### 4. Run the pipeline

**Run each step individually:**
```bash
# Step 1: Preprocess data and engineer features
python -m src.data_preprocessing

# Step 2: Train classical baselines
python -m src.classical_models

# Step 3: Train quantum models (VQC + Kernel SVM + Hybrid)
python -m src.quantum_models
```

**Or run the full pipeline at once:**
```bash
python -m src.evaluation
```

This runs everything end-to-end and saves plots, predictions, and comparison reports to `results/`.

## Requirements

- Python 3.10+
- Qiskit 2.x
- qiskit-machine-learning 0.9+
- scikit-learn, pandas, numpy, matplotlib

See `requirements.txt` for full list.

## Platform

Built using **IBM Qiskit** with local simulator (no IBM account required). All quantum circuits run on the `StatevectorSampler` for exact noiseless simulation.

## License

MIT
