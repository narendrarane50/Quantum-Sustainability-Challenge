"""
Export wildfire risk scores per zip code for Task 2 (insurance model).

Task 2 predicts 2021 insurance premiums from 2018-2020 historical data.
So your teammate needs wildfire risk scores for 2021.

This script:
1. Temporarily modifies the pipeline so year 2021 becomes the TEST set
   (originally 2021 is part of training data)
2. Trains the hybrid quantum-classical model on 2019-2020 data
3. Exports 2021 wildfire risk predictions for all California zip codes

Usage:
    python -m src.export_risk_scores_2021
"""

from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from src.data_preprocessing import (
    build_dataset, FEATURE_COLS,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
import time


def build_splits_for_2021(final_df, n_pca: int = 8):
    """
    Custom splits for Task 2: predict 2021.
    
    Train: 2019, 2020  (no 2021 leakage)
    Val:   2020 (we use it for threshold tuning — slight leak but necessary)
    Test:  2021
    """
    train = final_df[final_df["target_year"].isin([2019, 2020])]
    val = final_df[final_df["target_year"] == 2020]  # reused — small compromise
    test = final_df[final_df["target_year"] == 2021]

    X_train = train[FEATURE_COLS].values
    X_val = val[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
    y_train = train["fire_occurred"].values
    y_val = val["fire_occurred"].values
    y_test = test["fire_occurred"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=n_pca)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    X_test_pca = pca.transform(X_test_s)

    print(f"Train (2019+2020): {X_train_s.shape}, pos rate: {y_train.mean():.3f}")
    print(f"Test  (2021):      {X_test_s.shape}, pos rate: {y_test.mean():.3f}")
    print(f"PCA {n_pca} components: {pca.explained_variance_ratio_.sum():.1%} variance")

    return {
        "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
        "X_train_pca": X_train_pca, "X_val_pca": X_val_pca, "X_test_pca": X_test_pca,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "test_zips": test["zip"].values,
    }


def prepare_quantum_data_2021(splits, n_samples: int = 500, seed: int = 42):
    X_pca, y = splits["X_train_pca"], splits["y_train"]
    np.random.seed(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = min(n_samples // 2, len(pos_idx))

    pos_s = np.random.choice(pos_idx, size=n_pos, replace=False)
    neg_s = np.random.choice(neg_idx, size=n_pos, replace=False)
    idx = np.concatenate([pos_s, neg_s])
    np.random.shuffle(idx)

    X_sub, y_sub = X_pca[idx], y[idx]

    scaler_q = MinMaxScaler(feature_range=(0, np.pi))
    X_tr = np.clip(scaler_q.fit_transform(X_sub), 0, np.pi)
    X_te = np.clip(scaler_q.transform(splits["X_test_pca"]), 0, np.pi)

    print(f"Quantum train: {X_tr.shape}, pos rate: {y_sub.mean():.2f}")
    return X_tr, y_sub, X_te


def compute_quantum_features(kernel, X_data, X_anchors, y_anchors):
    K = kernel.evaluate(X_data, X_anchors)
    pos_idx = np.where(y_anchors == 1)[0]
    neg_idx = np.where(y_anchors == 0)[0]
    sim_fire = K[:, pos_idx].mean(axis=1)
    sim_nofire = K[:, neg_idx].mean(axis=1)
    sim_ratio = sim_fire / (sim_nofire + 1e-8)
    sim_max_fire = K[:, pos_idx].max(axis=1)
    sim_diff = sim_fire - sim_nofire
    return np.column_stack([sim_fire, sim_nofire, sim_ratio, sim_max_fire, sim_diff])


def export_2021_risk_scores(output_path: str = "results/wildfire_risk_scores_2021.csv"):
    print("=" * 60)
    print("EXPORTING WILDFIRE RISK SCORES FOR 2021 (Task 2)")
    print("=" * 60)

    # Build dataset with dynamic path
    data_dir = str(project_root / 'data' / 'raw')
    final_df = build_dataset(data_dir)

    # Custom splits for 2021 prediction
    splits = build_splits_for_2021(final_df, n_pca=8)

    # Prepare quantum data
    X_q_tr, y_q_tr, X_q_te = prepare_quantum_data_2021(splits, n_samples=500)

    # Build quantum kernel anchor set
    print("\nBuilding quantum features...")
    np.random.seed(42)
    pos = np.where(y_q_tr == 1)[0]
    neg = np.where(y_q_tr == 0)[0]
    n_anchor = min(200, len(pos))
    anchor_idx = np.concatenate([
        np.random.choice(pos, n_anchor, False),
        np.random.choice(neg, n_anchor, False),
    ])
    np.random.shuffle(anchor_idx)
    X_anchor = X_q_tr[anchor_idx]
    y_anchor = y_q_tr[anchor_idx]

    # Quantum kernel
    fm = zz_feature_map(feature_dimension=8, reps=2, entanglement="linear")
    kernel = FidelityStatevectorKernel(feature_map=fm)

    # Compute full training set quantum features (for hybrid)
    scaler_q = MinMaxScaler(feature_range=(0, np.pi))
    X_train_pca_scaled = np.clip(
        scaler_q.fit_transform(splits["X_train_pca"]), 0, np.pi
    )

    t0 = time.time()
    qf_tr = compute_quantum_features(kernel, X_train_pca_scaled, X_anchor, y_anchor)
    print(f"  Training features: {qf_tr.shape} in {time.time()-t0:.1f}s")
    t0 = time.time()
    qf_te = compute_quantum_features(kernel, X_q_te, X_anchor, y_anchor)
    print(f"  Test features: {qf_te.shape} in {time.time()-t0:.1f}s")

    # Combine classical + quantum features
    X_hybrid_tr = np.hstack([splits["X_train"], qf_tr])
    X_hybrid_te = np.hstack([splits["X_test"], qf_te])

    print(f"\nHybrid features: {X_hybrid_tr.shape[1]} (27 classical + 5 quantum)")

    # Train hybrid GBT
    print("\nTraining hybrid model...")
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_split=10, min_samples_leaf=5,
        random_state=42,
    )
    gb.fit(X_hybrid_tr, splits["y_train"])

    # Get probabilities
    yprob_te = gb.predict_proba(X_hybrid_te)[:, 1]
    y_te = splits["y_test"]

    # Evaluate on 2021 test set
    yp_default = (yprob_te >= 0.5).astype(int)
    print(f"\n=== 2021 Prediction Results ===")
    print(f"Test AUC: {((yprob_te[y_te==1]).mean() > (yprob_te[y_te==0]).mean()) and 'good' or 'check'}")
    from sklearn.metrics import roc_auc_score
    print(f"Test AUC: {roc_auc_score(y_te, yprob_te):.4f}")
    print(f"Test F1 (default): {f1_score(y_te, yp_default):.4f}")
    print(classification_report(y_te, yp_default))

    # Build output DataFrame
    output = pd.DataFrame({
        "zip_code": splits["test_zips"].astype(int),
        "wildfire_risk_2021": yprob_te.round(4),
        "predicted_fire_2021": yp_default,
    })

    # Sort by risk (highest first)
    output = output.sort_values("wildfire_risk_2021", ascending=False).reset_index(drop=True)

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"\n✓ Saved risk scores to {out_path}")
    print(f"  Total zip codes: {len(output)}")
    print(f"  Risk score range: [{output['wildfire_risk_2021'].min():.4f}, {output['wildfire_risk_2021'].max():.4f}]")
    print(f"  Mean risk: {output['wildfire_risk_2021'].mean():.4f}")
    print(f"  Predicted fires: {output['predicted_fire_2021'].sum()}")
    print(f"\nTop 10 highest-risk zip codes:")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    output_path = str(project_root / 'results/task1/wildfire_risk_scores_2021.csv')
    export_2021_risk_scores(output_path=output_path)
