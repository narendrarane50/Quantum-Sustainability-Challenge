# Methodology & Approach

## Task 1A: Quantum Algorithm for Wildfire Prediction

### Problem Formulation

We frame wildfire prediction as a **binary classification** problem: for each of ~2,600 California zip codes, predict whether at least one wildfire will occur in 2023 (1 = fire, 0 = no fire).

The dataset is heavily imbalanced — only ~5–8% of zip codes experience fires in any given year — making this a challenging classification task where recall on the minority class is critical.

### Data Pipeline

#### Raw Data Structure

The competition provides a single CSV containing two interleaved record types:

- **Fire incident records** (2,218 rows): Each row is a historical wildfire event with fields including agency, location (lat/lon, zip), start/containment dates, cause code, and area burned (GIS_ACRES).
- **Monthly weather observations** (~123,000 rows): Monthly temperature and precipitation data per zip code from weather stations.

#### Feature Engineering (20 features per zip code per year)

**Weather features (15)** — Aggregated from monthly station data:

| Feature | Description |
|---------|-------------|
| avg_tmax_annual | Mean daily high temperature (°C) |
| max_tmax / min_tmax | Hottest / coldest monthly avg high |
| tmax_range | Annual temperature range |
| avg_tmin_annual | Mean daily low temperature |
| max_tmin / min_tmin | Extremes of monthly avg low |
| total_precip | Annual total precipitation (mm) |
| avg_precip / max_precip | Monthly precipitation stats |
| dry_months | Count of months with <5mm rain |
| temp_std | Temperature variability |
| summer_avg_tmax/tmin | Jun–Sep average temperatures |
| summer_precip | Jun–Sep total rainfall |

**Fire history features (5)** — Cumulative stats from all prior years:

| Feature | Description |
|---------|-------------|
| hist_fire_count | Total number of past fires in this zip |
| hist_total_acres | Cumulative acres burned |
| hist_max_acres | Largest single fire |
| hist_avg_acres | Average fire size |
| hist_n_years_with_fire | Number of distinct years with fires |

#### Temporal Design

- **Lagged weather**: Weather from year Y-1 predicts fires in year Y (weather precedes fire risk).
- **Weather data availability**: Weather observations cover 2018–2021. For predicting 2022–2023, we use 2021 weather (most recent available).
- **No data leakage**: Fire history features only use data from years strictly before the prediction year.

### Dimensionality Reduction

Quantum simulators are computationally expensive, scaling as O(2^n) where n = number of qubits. We use **PCA** to reduce from 20 features to 4, retaining ~81% of variance. This maps directly to a 4-qubit circuit.

The PCA transformation is fit on the training set only, then applied to validation and test sets.

### Quantum Models

#### Model 1: Variational Quantum Classifier (VQC)

The VQC is a hybrid quantum-classical algorithm consisting of:

1. **Feature encoding** via ZZFeatureMap: Classical features x are encoded into quantum states using single-qubit rotations P(2xᵢ) and two-qubit entangling gates P(f(xᵢ, xⱼ)) that capture pairwise feature interactions.

2. **Parameterized ansatz** via RealAmplitudes: A trainable quantum circuit with Ry rotation gates whose angles θ are optimized during training.

3. **Measurement**: The final quantum state is measured, and the measurement outcome is mapped to a class label.

4. **Classical optimization**: COBYLA optimizer adjusts θ to minimize classification loss.

**Configuration:**
- 4 qubits, ZZFeatureMap (1 rep, linear entanglement)
- RealAmplitudes ansatz (2 reps, linear entanglement, 12 trainable parameters)
- Circuit depth: ~18
- COBYLA optimizer, 200 iterations

#### Model 2: Quantum Kernel SVM

Instead of optimizing a parameterized circuit, this approach uses the quantum feature map to compute a **kernel matrix**:

K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²

where φ(x) is the quantum state produced by the ZZFeatureMap. This kernel is fed to a classical SVM (with `class_weight='balanced'`).

**Configuration:**
- 4 qubits, ZZFeatureMap (2 reps, linear entanglement)
- Feature map depth: ~19
- SVM with precomputed kernel, C selected via grid search
- No iterative quantum optimization needed

#### Training Data for Quantum

Due to computational constraints of quantum simulation, we train on a **balanced subsample of 300 data points** (150 fire + 150 no-fire). Features are scaled to [0, π] for angle encoding in the quantum circuit.

### Resource Requirements

| Resource | VQC | Quantum Kernel SVM |
|----------|-----|-------------------|
| Qubits | 4 | 4 |
| Circuit depth | ~18 | ~19 |
| Trainable parameters | 12 | 0 |
| Training time (simulator) | ~40–130s | ~7s |
| Simulator memory | O(2⁴) = 16 amplitudes | O(2⁴) per kernel eval |
| Total circuit evaluations | ~200 × 300 = 60,000 | 300² + 2,593×300 = ~868,000 |

---

## Task 1B: Evaluation

### Classical Baselines

We train three classical models on the **full 20 features** using all 7,779 training samples:

1. **Logistic Regression** (balanced class weights)
2. **Random Forest** (200 trees, balanced weights)
3. **Gradient Boosting** (200 trees, max depth 4)
4. **Random Forest on 4 PCA features** (fair comparison with quantum)

### Comparison

Classical models significantly outperform quantum approaches on this dataset. Key factors:

1. **Data volume**: Classical models use 26× more training data (7,779 vs 300).
2. **Feature access**: Classical models use all 20 features; quantum uses 4 PCA components.
3. **Optimization**: Gradient-based classical optimizers converge reliably; the VQC's COBYLA optimizer struggled with the quantum loss landscape.

### Where Quantum Shows Promise

- **Quantum Kernel SVM achieved ~60% recall** on fire events, comparable to classical models' recall, though with lower precision.
- The quantum kernel maps data into a 2⁴ = 16-dimensional Hilbert space that may capture correlations inaccessible to classical polynomial kernels.
- With more qubits (handling 20+ features without PCA) and more training data, the gap could narrow.

### Limitations

- **Simulator-only**: No hardware noise or decoherence effects tested.
- **Small scale**: 4 qubits is well within classical simulability; quantum advantage typically requires ≥50+ qubits.
- **Tabular data**: QML advantages are more theorized for high-dimensional, structured data. Tabular data with engineered features tends to favor gradient-boosted trees.
- **No probability calibration**: VQC outputs are not calibrated probabilities, limiting threshold optimization.
