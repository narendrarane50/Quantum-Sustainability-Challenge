"""
Quantum machine learning models for wildfire prediction.

Implements three approaches:
1. Variational Quantum Classifier (VQC)
2. Quantum Kernel SVM
3. Hybrid Quantum-Classical model (quantum-enhanced features + GBT)
"""

import json
import time
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

from src.data_preprocessing import build_dataset, get_splits, prepare_quantum_data


# ──────────────────────────────────────────────
# VQC
# ──────────────────────────────────────────────

def train_vqc(X_tr, y_tr, num_qubits=8, fm_reps=1, ansatz_reps=2, max_iter=200, seed=42):
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=fm_reps, entanglement="linear")
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=ansatz_reps, entanglement="linear")
    circuit = feature_map.compose(ansatz)

    print(f"\n=== VQC Configuration ===")
    print(f"Qubits: {num_qubits}")
    print(f"Feature map: ZZFeatureMap (reps={fm_reps}, linear)")
    print(f"Ansatz: RealAmplitudes (reps={ansatz_reps}, linear)")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Trainable parameters: {ansatz.num_parameters}")
    print(f"\nCircuit diagram:")
    print(circuit.draw(output="text", fold=100))

    objective_values = []
    def callback(weights, obj_value):
        objective_values.append(float(obj_value))
        if len(objective_values) % 20 == 0:
            print(f"  Iteration {len(objective_values)}: loss = {obj_value:.4f}")

    np.random.seed(seed)
    initial_point = np.random.uniform(-0.1, 0.1, ansatz.num_parameters)

    vqc = VQC(
        sampler=StatevectorSampler(), feature_map=feature_map, ansatz=ansatz,
        optimizer=COBYLA(maxiter=max_iter), callback=callback, initial_point=initial_point,
    )

    # Subsample for VQC speed (use 200 balanced samples)
    np.random.seed(seed)
    pos = np.where(y_tr == 1)[0]
    neg = np.where(y_tr == 0)[0]
    n = min(100, len(pos))
    idx = np.concatenate([np.random.choice(pos, n, False), np.random.choice(neg, n, False)])
    np.random.shuffle(idx)

    print(f"\nTraining VQC on {len(idx)} samples...")
    t0 = time.time()
    vqc.fit(X_tr[idx], y_tr[idx])
    train_time = time.time() - t0
    print(f"Done in {train_time:.1f}s ({len(objective_values)} iterations)")

    return {
        "model": vqc, "objective_values": objective_values,
        "train_time": train_time, "num_qubits": num_qubits,
        "circuit_depth": circuit.depth(), "trainable_params": int(ansatz.num_parameters),
        "train_samples": len(idx),
        "config": {
            "feature_map": f"ZZFeatureMap(reps={fm_reps}, linear)",
            "ansatz": f"RealAmplitudes(reps={ansatz_reps}, linear)",
            "optimizer": f"COBYLA(maxiter={max_iter})",
        },
    }


# ──────────────────────────────────────────────
# Quantum Kernel SVM
# ──────────────────────────────────────────────

def train_quantum_kernel_svm(X_tr, y_tr, X_va, y_va, X_te, num_qubits=8, fm_reps=2):
    fm = zz_feature_map(feature_dimension=num_qubits, reps=fm_reps, entanglement="linear")
    kernel = FidelityStatevectorKernel(feature_map=fm)

    print(f"\n=== Quantum Kernel SVM ===")
    print(f"Feature map: ZZFeatureMap ({num_qubits} qubits, reps={fm_reps}, linear)")
    print(f"Depth: {fm.depth()}")

    print("\nComputing kernels...", flush=True)
    t0 = time.time()
    K_train = kernel.evaluate(X_tr)
    print(f"  K_train {K_train.shape}: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    K_val = kernel.evaluate(X_va, X_tr)
    print(f"  K_val   {K_val.shape}: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    K_test = kernel.evaluate(X_te, X_tr)
    print(f"  K_test  {K_test.shape}: {time.time()-t0:.1f}s", flush=True)

    best_f1, best_C, best_svc = 0, None, None
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        svc = SVC(kernel="precomputed", class_weight="balanced", C=C)
        svc.fit(K_train, y_tr)
        yp = svc.predict(K_val)
        f1 = f1_score(y_va, yp, zero_division=0)
        print(f"  C={C:5.1f}: F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_C, best_svc = f1, C, svc

    return {
        "model": best_svc, "kernel": kernel,
        "K_train": K_train, "K_val": K_val, "K_test": K_test,
        "best_C": best_C, "val_f1": best_f1,
        "num_qubits": num_qubits, "feature_map_depth": fm.depth(),
        "train_samples": len(X_tr),
    }


# ──────────────────────────────────────────────
# Hybrid Quantum-Classical Model
# ──────────────────────────────────────────────

def compute_quantum_features(kernel, X_data, X_anchors, y_anchors):
    """Compute quantum similarity features relative to fire/no-fire anchor points."""
    K = kernel.evaluate(X_data, X_anchors)
    pos_idx = np.where(y_anchors == 1)[0]
    neg_idx = np.where(y_anchors == 0)[0]

    sim_fire = K[:, pos_idx].mean(axis=1)
    sim_nofire = K[:, neg_idx].mean(axis=1)
    sim_ratio = sim_fire / (sim_nofire + 1e-8)
    sim_max_fire = K[:, pos_idx].max(axis=1)
    sim_diff = sim_fire - sim_nofire

    return np.column_stack([sim_fire, sim_nofire, sim_ratio, sim_max_fire, sim_diff])


def train_hybrid_model(splits, X_q_tr, y_q_tr, X_q_va, X_q_te, X_q_tr_full, num_qubits=8):
    """
    Hybrid model: classical features + quantum kernel similarity features,
    fed into a Gradient Boosting classifier.
    """
    print("\n=== Hybrid Quantum-Classical Model ===")

    # Build quantum kernel on anchor set
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

    fm = zz_feature_map(feature_dimension=num_qubits, reps=2, entanglement="linear")
    kernel = FidelityStatevectorKernel(feature_map=fm)

    print(f"Anchor set: {X_anchor.shape}")
    print("Computing quantum features...", flush=True)

    t0 = time.time()
    qf_tr = compute_quantum_features(kernel, X_q_tr_full, X_anchor, y_anchor)
    print(f"  Training features: {qf_tr.shape} in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    qf_va = compute_quantum_features(kernel, X_q_va, X_anchor, y_anchor)
    print(f"  Validation features: {qf_va.shape} in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    qf_te = compute_quantum_features(kernel, X_q_te, X_anchor, y_anchor)
    print(f"  Test features: {qf_te.shape} in {time.time()-t0:.1f}s", flush=True)

    # Combine classical + quantum features
    X_hybrid_tr = np.hstack([splits["X_train"], qf_tr])
    X_hybrid_va = np.hstack([splits["X_val"], qf_va])
    X_hybrid_te = np.hstack([splits["X_test"], qf_te])

    n_classical = splits["X_train"].shape[1]
    n_quantum = qf_tr.shape[1]
    print(f"\nHybrid features: {X_hybrid_tr.shape[1]} ({n_classical} classical + {n_quantum} quantum)")

    # Train hybrid GBT
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_split=10, min_samples_leaf=5,
        random_state=42,
    )
    gb.fit(X_hybrid_tr, splits["y_train"])

    return {
        "model": gb,
        "X_hybrid_tr": X_hybrid_tr, "X_hybrid_va": X_hybrid_va, "X_hybrid_te": X_hybrid_te,
        "quantum_feature_names": ["q_sim_fire", "q_sim_nofire", "q_sim_ratio", "q_max_sim_fire", "q_sim_diff"],
        "n_classical": n_classical, "n_quantum": n_quantum,
    }


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def evaluate_vqc(vqc_result, X_va, y_va, X_te, y_te):
    model = vqc_result["model"]
    print("\n=== VQC — Validation (2022) ===")
    yp_va = model.predict(X_va)
    print(classification_report(y_va, yp_va))
    print("=== VQC — Test (2023) ===")
    yp_te = model.predict(X_te)
    print(classification_report(y_te, yp_te))
    print(f"Confusion matrix:\n{confusion_matrix(y_te, yp_te)}")
    return {
        "val_f1": f1_score(y_va, yp_va, zero_division=0),
        "test_f1": f1_score(y_te, yp_te, zero_division=0),
        "test_predictions": yp_te,
    }


def evaluate_qkernel(qk_result, y_va, y_te):
    model = qk_result["model"]
    print("\n=== Quantum Kernel SVM — Validation (2022) ===")
    yp_va = model.predict(qk_result["K_val"])
    print(classification_report(y_va, yp_va))
    print("=== Quantum Kernel SVM — Test (2023) ===")
    yp_te = model.predict(qk_result["K_test"])
    print(classification_report(y_te, yp_te))
    print(f"Confusion matrix:\n{confusion_matrix(y_te, yp_te)}")
    return {
        "val_f1": f1_score(y_va, yp_va, zero_division=0),
        "test_f1": f1_score(y_te, yp_te, zero_division=0),
        "test_predictions": yp_te,
    }


def evaluate_hybrid(hybrid_result, y_va, y_te):
    model = hybrid_result["model"]
    X_va = hybrid_result["X_hybrid_va"]
    X_te = hybrid_result["X_hybrid_te"]

    print("\n=== Hybrid Model — Validation (2022) ===")
    yp_va = model.predict(X_va)
    print(classification_report(y_va, yp_va))

    print("=== Hybrid Model — Test (2023) ===")
    yp_te = model.predict(X_te)
    yprob_te = model.predict_proba(X_te)[:, 1]
    print(classification_report(y_te, yp_te))
    print(f"Test F1:  {f1_score(y_te, yp_te):.4f}")
    print(f"Test AUC: {roc_auc_score(y_te, yprob_te):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_te, yp_te)}")

    # Feature importance
    from src.data_preprocessing import FEATURE_COLS
    feat_names = list(FEATURE_COLS) + hybrid_result["quantum_feature_names"]
    importances = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop 15 feature importances:")
    for name, imp in importances[:15]:
        marker = " <<< QUANTUM" if name.startswith("q_") else ""
        print(f"  {name:25s}: {imp:.4f}{marker}")

    return {
        "val_f1": f1_score(y_va, yp_va),
        "test_f1": f1_score(y_te, yp_te),
        "test_auc": roc_auc_score(y_te, yprob_te),
        "test_predictions": yp_te,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    final_df = build_dataset("data/raw")
    splits = get_splits(final_df, n_pca=8)
    X_q_tr, y_q_tr, X_q_va, X_q_te, X_q_tr_full, _ = prepare_quantum_data(splits, n_samples=500)
    y_va, y_te = splits["y_val"], splits["y_test"]

    # VQC
    print("\n" + "=" * 50)
    print("TRAINING VQC")
    print("=" * 50)
    vqc_result = train_vqc(X_q_tr, y_q_tr, num_qubits=8)
    vqc_eval = evaluate_vqc(vqc_result, X_q_va, y_va, X_q_te, y_te)

    # Quantum Kernel SVM
    print("\n" + "=" * 50)
    print("TRAINING QUANTUM KERNEL SVM")
    print("=" * 50)
    qk_result = train_quantum_kernel_svm(X_q_tr, y_q_tr, X_q_va, y_va, X_q_te, num_qubits=8)
    qk_eval = evaluate_qkernel(qk_result, y_va, y_te)

    # Hybrid Model
    print("\n" + "=" * 50)
    print("TRAINING HYBRID QUANTUM-CLASSICAL MODEL")
    print("=" * 50)
    hybrid_result = train_hybrid_model(splits, X_q_tr, y_q_tr, X_q_va, X_q_te, X_q_tr_full, num_qubits=8)
    hybrid_eval = evaluate_hybrid(hybrid_result, y_va, y_te)

    # Save
    Path("results").mkdir(exist_ok=True)
    summary = {
        "VQC": {"val_f1": vqc_eval["val_f1"], "test_f1": vqc_eval["test_f1"],
                "qubits": 8, **vqc_result["config"]},
        "Quantum Kernel SVM": {"val_f1": qk_eval["val_f1"], "test_f1": qk_eval["test_f1"],
                               "qubits": 8, "best_C": qk_result["best_C"]},
        "Hybrid Model": {"val_f1": hybrid_eval["val_f1"], "test_f1": hybrid_eval["test_f1"],
                         "test_auc": hybrid_eval["test_auc"], "qubits": 8},
    }
    with open("results/quantum_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to results/quantum_results.json")
