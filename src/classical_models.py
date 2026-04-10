"""
Classical baseline models for wildfire prediction.
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import build_dataset, get_splits, FEATURE_COLS


def train_classical_baselines(splits):
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_va, y_va = splits["X_val"], splits["y_val"]
    X_tr_pca, X_va_pca = splits["X_train_pca"], splits["X_val_pca"]
    results = {}

    for name, model, use_pca in [
        ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), False),
        ("Random Forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1), False),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42), False),
        ("RF (8 PCA)", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1), True),
    ]:
        X, Xv = (X_tr_pca, X_va_pca) if use_pca else (X_tr, X_va)
        t0 = time.time()
        model.fit(X, y_tr)
        dt = time.time() - t0
        y_prob = model.predict_proba(Xv)[:, 1]
        y_pred = model.predict(Xv)
        feat_imp = None
        if hasattr(model, "feature_importances_") and not use_pca:
            feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        results[name] = {
            "model": model, "features": 8 if use_pca else len(FEATURE_COLS),
            "train_time": dt, "val_f1": f1_score(y_va, y_pred),
            "val_auc": roc_auc_score(y_va, y_prob), "use_pca": use_pca,
        }
        if feat_imp:
            results[name]["feature_importances"] = feat_imp

    return results


def print_summary(results):
    print(f"\n{'Model':<25} {'Feat':>5} {'F1':>8} {'AUC':>8} {'Time':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['features']:>5} {r['val_f1']:>8.4f} {r['val_auc']:>8.4f} {r['train_time']:>7.1f}s")


def evaluate_on_test(results, splits):
    X_te, y_te = splits["X_test"], splits["y_test"]
    X_te_pca = splits["X_test_pca"]
    print(f"\n{'Model':<25} {'Feat':>5} {'F1':>8} {'AUC':>8}")
    print("-" * 50)
    for name, r in results.items():
        X = X_te_pca if r.get("use_pca") else X_te
        y_pred = r["model"].predict(X)
        y_prob = r["model"].predict_proba(X)[:, 1]
        print(f"{name:<25} {r['features']:>5} {f1_score(y_te, y_pred):>8.4f} {roc_auc_score(y_te, y_prob):>8.4f}")


if __name__ == "__main__":
    final_df = build_dataset("data/raw")
    splits = get_splits(final_df)
    results = train_classical_baselines(splits)
    print("\n=== VALIDATION ===")
    print_summary(results)
    print("\n=== TEST ===")
    evaluate_on_test(results, splits)
    Path("results/task1").mkdir(parents=True, exist_ok=True)
    out = {n: {k: v for k, v in r.items() if k != "model"} for n, r in results.items()}
    with open("results/task1/classical_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved to results/task1/classical_results.json")
