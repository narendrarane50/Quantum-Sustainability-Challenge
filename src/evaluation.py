"""
Full pipeline evaluation: classical baselines + quantum models + comparison.
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import build_dataset, get_splits, prepare_quantum_data, FEATURE_COLS
from src.classical_models import train_classical_baselines
from src.quantum_models import (
    train_vqc, train_quantum_kernel_svm, train_hybrid_model,
    evaluate_vqc, evaluate_qkernel, evaluate_hybrid,
)


def run_full_evaluation():
    """Run all models and generate comparison report."""

    # ── Data ──
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    final_df = build_dataset("data/raw")
    splits = get_splits(final_df, n_pca=8)
    X_q_tr, y_q_tr, X_q_va, X_q_te, X_q_tr_full, _ = prepare_quantum_data(splits, n_samples=500)
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
    vqc_result = train_vqc(X_q_tr, y_q_tr, num_qubits=8, max_iter=200)
    vqc_eval = evaluate_vqc(vqc_result, X_q_va, y_va, X_q_te, y_te)

    qk_result = train_quantum_kernel_svm(X_q_tr, y_q_tr, X_q_va, y_va, X_q_te, num_qubits=8)
    qk_eval = evaluate_qkernel(qk_result, y_va, y_te)

    hybrid_result = train_hybrid_model(splits, X_q_tr, y_q_tr, X_q_va, X_q_te, X_q_tr_full, num_qubits=8)
    hybrid_eval = evaluate_hybrid(hybrid_result, y_va, y_te)

    # ── Final comparison ──
    print("\n" + "=" * 60)
    print("FINAL COMPARISON — TEST SET (2023)")
    print("=" * 60)

    all_results = {}
    for name, r in classical.items():
        X = splits["X_test_pca"] if r.get("use_pca") else splits["X_test"]
        yp = r["model"].predict(X)
        yprob = r["model"].predict_proba(X)[:, 1]
        all_results[name] = {"type": "Classical", "f1": f1_score(y_te, yp), "auc": roc_auc_score(y_te, yprob)}

    all_results["VQC"] = {"type": "Quantum", "f1": vqc_eval["test_f1"], "auc": None}
    all_results["Quantum Kernel SVM"] = {"type": "Quantum", "f1": qk_eval["test_f1"], "auc": None}
    all_results["Hybrid (Classical + Quantum)"] = {
        "type": "Hybrid", "f1": hybrid_eval["test_f1"], "auc": hybrid_eval["test_auc"]
    }

    print(f"\n{'Model':<30} {'Type':<10} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "—"
        print(f"{name:<30} {r['type']:<10} {r['f1']:>8.4f} {auc_str:>8}")

    # ── Save outputs ──
    results_dir = Path("results/task1")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    names = list(all_results.keys())
    f1s = [r["f1"] for r in all_results.values()]
    colors = {"Classical": "#534AB7", "Quantum": "#D85A30", "Hybrid": "#1D9E75"}
    bar_colors = [colors[r["type"]] for r in all_results.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1s, color=bar_colors)
    ax.set_xlabel("F1 Score")
    ax.set_title("Test Set Performance — 2023 Wildfire Prediction")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{f1:.3f}", va="center")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=c) for c in colors.values()], colors.keys(), loc="lower right")
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved {results_dir / 'model_comparison.png'}")

    # Convergence plot
    if vqc_result["objective_values"]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(vqc_result["objective_values"], color="#534AB7", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value (Loss)")
        ax.set_title("VQC Training Convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "vqc_convergence.png", dpi=150)
        plt.close()
        print(f"Saved {results_dir / 'vqc_convergence.png'}")

    # Predictions
    predictions = pd.DataFrame({
        "zip_code": splits["test_zips"].astype(int),
        "predicted_fire_2023": hybrid_eval["test_predictions"],
        "actual_fire_2023": y_te,
    })
    predictions.to_csv(results_dir / "wildfire_predictions_2023.csv", index=False)
    print(f"\nPredictions saved to {results_dir / 'wildfire_predictions_2023.csv'}")
    print(f"  Predicted fires: {predictions['predicted_fire_2023'].sum()}")
    print(f"  Actual fires: {predictions['actual_fire_2023'].sum()}")

    # Summary JSON
    with open(results_dir / "comparison_summary.json", "w") as f:
        json.dump({n: {k: v for k, v in r.items()} for n, r in all_results.items()}, f, indent=2, default=str)
    print(f"Summary saved to {results_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    run_full_evaluation()
