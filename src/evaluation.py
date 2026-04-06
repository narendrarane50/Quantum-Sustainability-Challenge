"""
Evaluation and comparison of classical vs quantum models.

Generates comparison tables, plots, and the final predictions CSV.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)

from src.data_preprocessing import (
    build_dataset, get_splits, prepare_quantum_data, FEATURE_COLS,
)
from src.classical_models import train_classical_baselines
from src.quantum_models import (
    train_vqc, train_quantum_kernel_svm,
    evaluate_vqc, evaluate_qkernel,
)


def run_full_evaluation():
    """Run all models and generate comparison report."""

    # ── Data ──
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    final_df = build_dataset("data/raw")
    splits = get_splits(final_df, n_pca=4)
    X_tr_q, y_tr_q, X_va_q, X_te_q, _ = prepare_quantum_data(splits, n_samples=300)
    y_va, y_te = splits["y_val"], splits["y_test"]

    # ── Classical ──
    print("\n" + "=" * 60)
    print("STEP 2: CLASSICAL BASELINES")
    print("=" * 60)
    classical = train_classical_baselines(splits)

    # ── Quantum ──
    print("\n" + "=" * 60)
    print("STEP 3: QUANTUM MODELS")
    print("=" * 60)
    vqc_result = train_vqc(X_tr_q, y_tr_q, num_qubits=4, max_iter=200)
    vqc_eval = evaluate_vqc(vqc_result, X_va_q, y_va, X_te_q, y_te)

    qk_result = train_quantum_kernel_svm(
        X_tr_q, y_tr_q, X_va_q, y_va, X_te_q, num_qubits=4
    )
    qk_eval = evaluate_qkernel(qk_result, y_va, y_te)

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("STEP 4: COMPARISON")
    print("=" * 60)

    # Test set results for classical models
    test_results = {}
    for name, r in classical.items():
        model = r["model"]
        X = splits["X_test_pca"] if r["features"] == 4 else splits["X_test"]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        test_results[name] = {
            "type": "Classical",
            "features": r["features"],
            "test_f1": f1_score(y_te, y_pred),
            "test_auc": roc_auc_score(y_te, y_prob),
        }

    test_results["VQC"] = {
        "type": "Quantum", "features": 4,
        "test_f1": vqc_eval["test_f1"], "test_auc": None,
    }
    test_results["Quantum Kernel SVM"] = {
        "type": "Quantum", "features": 4,
        "test_f1": qk_eval["test_f1"], "test_auc": None,
    }

    print(f"\n{'Model':<25} {'Type':<10} {'Feat':>5} {'Test F1':>9} {'Test AUC':>10}")
    print("-" * 62)
    for name, r in test_results.items():
        auc_str = f"{r['test_auc']:.4f}" if r["test_auc"] is not None else "—"
        print(f"{name:<25} {r['type']:<10} {r['features']:>5} {r['test_f1']:>9.4f} {auc_str:>10}")

    # ── Plots ──
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    plot_comparison(test_results, results_dir)
    plot_convergence(vqc_result["objective_values"], results_dir)
    plot_feature_importance(classical, results_dir)

    # ── Predictions ──
    predictions = pd.DataFrame({
        "zip_code": splits["test_zips"].astype(int),
        "predicted_fire_2023": qk_eval["test_predictions"],
        "actual_fire_2023": y_te,
    })
    predictions.to_csv(results_dir / "wildfire_predictions_2023.csv", index=False)
    print(f"\nPredictions saved to {results_dir / 'wildfire_predictions_2023.csv'}")
    print(f"  Predicted fires: {predictions['predicted_fire_2023'].sum()}")
    print(f"  Actual fires: {predictions['actual_fire_2023'].sum()}")

    # ── Save summary ──
    summary = {name: {k: v for k, v in r.items()} for name, r in test_results.items()}
    with open(results_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {results_dir / 'comparison_summary.json'}")


def plot_comparison(test_results: dict, out_dir: Path):
    """Bar chart comparing F1 scores across all models."""
    names = list(test_results.keys())
    f1s = [r["test_f1"] for r in test_results.values()]
    colors = ["#534AB7" if r["type"] == "Classical" else "#D85A30"
              for r in test_results.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1s, color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_title("Test Set Performance — 2023 Wildfire Prediction")
    ax.set_xlim(0, max(f1s) * 1.3)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}", va="center", fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(color="#534AB7"), Patch(color="#D85A30")],
        ["Classical", "Quantum"],
        loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'model_comparison.png'}")


def plot_convergence(objective_values: list, out_dir: Path):
    """Plot VQC training convergence."""
    if not objective_values:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(objective_values, color="#534AB7", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value (Loss)")
    ax.set_title("VQC Training Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "vqc_convergence.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'vqc_convergence.png'}")


def plot_feature_importance(classical_results: dict, out_dir: Path):
    """Plot Random Forest feature importances."""
    rf_result = classical_results.get("Random Forest")
    if rf_result is None or "feature_importances" not in rf_result:
        return

    imp = pd.Series(rf_result["feature_importances"]).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color="#D85A30")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importances.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'feature_importances.png'}")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_full_evaluation()
