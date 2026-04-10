"""
Task 2 — Insurance Premium Forecaster v2
Deloitte Quantum Sustainability Challenge 2026

Three-way comparison on REAL data:
  A) Classical Baseline  — organizer 'Avg Fire Risk Score' (per row, varies by category)
  B) Quantum ML          — Task 1 hybrid quantum model probability (per ZIP, 0-1)
  C) Hybrid Synergy      — both signals combined

Pipeline:
  Train:    2018 + 2019
  Validate: 2020  (model selection)
  Predict:  2021  (final output)

Target: log1p(Premium_Rate) where rate = Earned Premium / Earned Exposure

Usage:
  python premium_forecaster_v2.py \
      --data  cal_insurance_fire_census_weather.csv \
      --risk  wildfire_risk_scores_2021.csv \
      --out   results/v2/
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PAL = dict(
    blue  = "#1A3C5E",
    coral = "#E05C3A",
    teal  = "#1D7874",
    gold  = "#D4A017",
    slate = "#4A6280",
    light = "#EFF4F9",
    grid  = "#D0DCE8",
    text  = "#1A2E44",
)
MODEL_COLORS = {
    "A — Organizer Risk Score": PAL["slate"],
    "B — Quantum ML (Task 1)":  PAL["coral"],
    "C — Hybrid Synergy":       PAL["teal"],
}


# =============================================================================
# 1. LOAD
# =============================================================================

def load_data(data_path: str, risk_path: str) -> pd.DataFrame:
    log.info("Loading insurance CSV ...")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [c.strip().replace("  ", " ") for c in df.columns]
    log.info(f"  {df.shape[0]:,} rows x {df.shape[1]} cols | years: "
             f"{sorted(df['Year'].dropna().unique().tolist())}")

    log.info("Loading Task 1 quantum risk scores ...")
    risk = pd.read_csv(risk_path)
    zip_col  = next(c for c in risk.columns if "zip"  in c.lower())
    prob_col = next(c for c in risk.columns
                    if "risk" in c.lower() and
                    ("prob" in c.lower() or "2021" in c.lower()))
    risk = (risk[[zip_col, prob_col]]
            .rename(columns={zip_col: "ZIP", prob_col: "quantum_risk_prob"}))
    log.info(f"  {len(risk):,} ZIPs | range "
             f"[{risk['quantum_risk_prob'].min():.4f}, "
             f"{risk['quantum_risk_prob'].max():.4f}]")

    df = df.merge(risk, on="ZIP", how="left")
    n_miss = df["quantum_risk_prob"].isna().sum()
    med_q  = df["quantum_risk_prob"].median()
    df["quantum_risk_prob"] = df["quantum_risk_prob"].fillna(med_q)
    if n_miss:
        log.info(f"  {n_miss} rows had no Task 1 score - filled with median {med_q:.4f}")
    return df


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def engineer(df: pd.DataFrame):
    log.info("Engineering features ...")

    # Numeric coercion
    for c in ["Earned Premium", "Earned Exposure", "Avg Fire Risk Score",
              "Year", "quantum_risk_prob",
              "Cov A Amount Weighted Avg", "Cov C Amount Weighted Avg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = [c for c in df.columns if c.startswith("Category_")]
    df["Active_Category"] = df[cat_cols].astype(float).idxmax(axis=1)

    # Target: Premium Rate — zero-exposure rows get rate = 0
    df["Premium_Rate"] = np.where(
        df["Earned Exposure"] > 0,
        df["Earned Premium"] / df["Earned Exposure"],
        0.0,
    )
    # Clip per-category outliers (top 0.5%)
    for cat in df["Active_Category"].unique():
        mask = df["Active_Category"] == cat
        cap  = df.loc[mask, "Premium_Rate"].quantile(0.995)
        df.loc[mask, "Premium_Rate"] = df.loc[mask, "Premium_Rate"].clip(upper=cap)

    df["Premium_Rate"] = (df["Premium_Rate"]
                          .replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0))
    df["log_Rate"] = np.log1p(df["Premium_Rate"])

    # Loss ratio: sum all incurred loss columns
    loss_cols = [c for c in df.columns if "Incurred" in c]
    df["Total_Incurred"] = (df[loss_cols]
                            .apply(pd.to_numeric, errors="coerce")
                            .fillna(0).sum(axis=1))
    df["Loss_Ratio"] = (
        df["Total_Incurred"] / df["Earned Premium"].clip(lower=1)
    ).clip(upper=10).fillna(0)

    # Normalise organizer score to [0,1]  (raw range is -0.21 to 4.0)
    frs = df["Avg Fire Risk Score"].clip(lower=0)
    df["Org_Risk_Norm"] = frs / (frs.max() + 1e-9)

    # Quantum x category interaction terms
    df["q_x_HO"]  = df["quantum_risk_prob"] * df["Category_HO"].astype(float)
    df["q_x_RT"]  = df["quantum_risk_prob"] * df["Category_RT"].astype(float)
    df["q_excess"] = df["quantum_risk_prob"] - df["quantum_risk_prob"].mean()

    # Grouped temporal lags (ZIP x Category x Year)
    df = df.sort_values(["ZIP", "Active_Category", "Year"]).reset_index(drop=True)
    grp = df.groupby(["ZIP", "Active_Category"])

    df["log_Rate_Lag1"]   = grp["log_Rate"].shift(1)
    df["log_Rate_Lag2"]   = grp["log_Rate"].shift(2)
    df["Rate_YoY_Delta"]  = grp["log_Rate"].diff()
    df["Exposure_Lag1"]   = grp["Earned Exposure"].shift(1)
    df["Loss_Ratio_Lag1"] = grp["Loss_Ratio"].shift(1)
    df["Org_Risk_Lag1"]   = grp["Org_Risk_Norm"].shift(1)   # lagged - no leakage

    # Kill any inf that crept in from log(0) neighbours after shift
    for c in ["log_Rate_Lag1", "log_Rate_Lag2", "Rate_YoY_Delta"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # Census features
    census = ["total_population", "median_income",
              "housing_value", "housing_vacancy_number"]
    for c in census:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    df = df.dropna(subset=["log_Rate_Lag1"]).copy()
    log.info(f"  Post-engineering: {df.shape[0]:,} rows")

    avail_census = [c for c in census if c in df.columns]
    q_inter      = ["q_x_HO", "q_x_RT", "q_excess"]
    return df, cat_cols, avail_census, q_inter


# =============================================================================
# 3. FEATURE SETS
# =============================================================================

def build_feature_sets(df, cat_cols, census_cols, q_inter_cols):
    temporal = ["log_Rate_Lag1", "log_Rate_Lag2", "Rate_YoY_Delta",
                "Exposure_Lag1", "Loss_Ratio_Lag1"]
    cov_cols = [c for c in ["Cov A Amount Weighted Avg", "Cov C Amount Weighted Avg"]
                if c in df.columns]

    A = temporal + ["Org_Risk_Lag1"]                              + census_cols + cov_cols + cat_cols
    B = temporal + ["quantum_risk_prob"] + q_inter_cols           + census_cols + cov_cols + cat_cols
    C = temporal + ["Org_Risk_Lag1", "quantum_risk_prob"] + q_inter_cols + census_cols + cov_cols + cat_cols

    def dedup(feats):
        seen, out = set(), []
        for f in feats:
            if f in df.columns and f not in seen:
                out.append(f); seen.add(f)
        return out

    return {
        "A — Organizer Risk Score": dedup(A),
        "B — Quantum ML (Task 1)":  dedup(B),
        "C — Hybrid Synergy":       dedup(C),
    }


# =============================================================================
# 4. TRAIN & EVALUATE
# =============================================================================

def make_gbm():
    return GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=4,
        subsample=0.75, min_samples_leaf=8, max_features=0.7,
        random_state=42,
    )


def calc_metrics(true_rate, pred_rate):
    mae  = mean_absolute_error(true_rate, pred_rate)
    rmse = np.sqrt(mean_squared_error(true_rate, pred_rate))
    r2   = r2_score(true_rate, pred_rate)
    mask = true_rate > 10
    mape = (np.mean(np.abs((true_rate[mask] - pred_rate[mask]) /
                            true_rate[mask])) * 100) if mask.sum() else np.nan
    return mae, rmse, r2, mape


def train_all(df, feature_sets):
    train_df = df[df["Year"].isin([2018, 2019])]
    val_df   = df[df["Year"] == 2020]
    test_df  = df[df["Year"] == 2021]
    target   = "log_Rate"

    results = {}
    log.info("\n--- A / B / C model training (validation year = 2020) ---")

    for name, feats in feature_sets.items():
        train_med = train_df[feats].median()

        X_tr = train_df[feats].fillna(train_med)
        y_tr = train_df[target]
        X_va = val_df[feats].fillna(train_med)
        y_va = val_df[target]
        X_te = test_df[feats].fillna(train_med)

        model = make_gbm()
        model.fit(X_tr, y_tr)

        pred_log_va  = model.predict(X_va)
        pred_rate_va = np.expm1(pred_log_va)
        true_rate_va = np.expm1(y_va.values)
        mae, rmse, r2, mape = calc_metrics(true_rate_va, pred_rate_va)

        log.info(f"  {name:<30}  MAE={mae:>8,.1f}  RMSE={rmse:>9,.1f}"
                 f"  R2={r2:.4f}  MAPE={mape:.1f}%")

        results[name] = dict(
            model       = model,
            feats       = feats,
            train_med   = train_med,
            val_mae     = mae,
            val_rmse    = rmse,
            val_r2      = r2,
            val_mape    = mape,
            pred_log_va = pred_log_va,
            true_log_va = y_va.values,
            pred_log_te = model.predict(X_te),
            val_df      = val_df.copy(),
            test_df     = test_df.copy(),
        )

    return results


# =============================================================================
# 5. SAVE CSV OUTPUTS
# =============================================================================

def save_outputs(results, best_name, out_path):
    res  = results[best_name]
    test = res["test_df"].copy()
    test["Predicted_Rate"]           = np.expm1(res["pred_log_te"]).clip(0)
    test["Predicted_Earned_Premium"] = (
        test["Predicted_Rate"] * test["Earned Exposure"].clip(lower=0)
    ).clip(lower=0)

    # Per-category detail CSV
    detail = test[["ZIP", "Active_Category", "Earned Exposure",
                   "Predicted_Rate", "Predicted_Earned_Premium"]].copy()
    detail.to_csv(out_path / "task2_2021_predictions_by_category.csv", index=False)

    # Aggregated per-ZIP CSV
    zip_agg = (
        test.groupby("ZIP")
            .agg(
                Predicted_Earned_Premium=("Predicted_Earned_Premium", "sum"),
                Quantum_Risk_Prob       =("quantum_risk_prob", "first"),
                Avg_Org_Fire_Risk_Score =("Avg Fire Risk Score", "mean"),
            )
            .reset_index()
            .sort_values("Predicted_Earned_Premium", ascending=False)
    )
    zip_agg.to_csv(out_path / "task2_2021_predictions_by_zip.csv", index=False)

    total = zip_agg["Predicted_Earned_Premium"].sum()
    log.info(f"  Per-category rows : {len(detail):,}")
    log.info(f"  Unique ZIP codes  : {len(zip_agg):,}")
    log.info(f"  Total 2021 premium: ${total:,.0f}")
    return detail, zip_agg


# =============================================================================
# 6. VISUALISATIONS
# =============================================================================

def _spine(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor(PAL["grid"]); sp.set_linewidth(0.8)
    ax.tick_params(colors=PAL["text"], labelsize=9)


def fig1_actual_vs_predicted(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="white")
    fig.suptitle(
        "Actual vs Predicted Premium Rate — 2020 Validation Set\n"
        "A: Organizer Score  |  B: Quantum ML (Task 1)  |  C: Hybrid",
        fontsize=13, fontweight="bold", color=PAL["text"], y=1.02,
    )
    for ax, (name, res) in zip(axes, results.items()):
        true_v = np.expm1(res["true_log_va"]).clip(1, None)
        pred_v = np.expm1(res["pred_log_va"]).clip(1, None)
        hb = ax.hexbin(true_v, pred_v, gridsize=38, cmap="Blues",
                       mincnt=1, linewidths=0.2, xscale="log", yscale="log")
        fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.82).set_label("# policies", fontsize=8)
        lo = min(true_v.min(), pred_v.min()) * 0.9
        hi = max(true_v.max(), pred_v.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color=PAL["coral"], lw=2, ls="--")
        ax.set(xlim=(lo, hi), ylim=(lo, hi),
               xlabel="Actual Rate ($/policy)", ylabel="Predicted Rate ($/policy)")
        ax.set_title(name, fontsize=10, fontweight="bold",
                     color=MODEL_COLORS[name], pad=8)
        ax.text(0.05, 0.93,
                f"R2 = {res['val_r2']:.3f}\nMAPE = {res['val_mape']:.1f}%",
                transform=ax.transAxes, fontsize=9.5, color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=MODEL_COLORS[name], alpha=0.88))
        ax.set_facecolor(PAL["light"])
        ax.grid(True, color=PAL["grid"], lw=0.4)
        _spine(ax)

    plt.tight_layout()
    plt.savefig(out_path / "fig1_actual_vs_predicted.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved fig1_actual_vs_predicted.png")


def fig2_model_comparison(results, out_path):
    names  = list(results.keys())
    labels = ["MAE ($/policy)", "RMSE ($/policy)", "R2 Score", "MAPE (%)"]
    keys   = ["val_mae", "val_rmse", "val_r2", "val_mape"]
    higher_better = {"val_r2"}

    fig, axes = plt.subplots(1, 4, figsize=(17, 5), facecolor="white")
    fig.suptitle(
        "A vs B vs C — Model Comparison on 2020 Validation\n"
        "Lower MAE / RMSE / MAPE is better  |  Higher R2 is better",
        fontsize=13, fontweight="bold", color=PAL["text"],
    )
    for ax, label, key in zip(axes, labels, keys):
        vals = [results[n][key] for n in names]
        best = (max if key in higher_better else min)(vals)
        bars = ax.barh(names, vals, color=[MODEL_COLORS[n] for n in names],
                       height=0.5, edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=10, fontweight="bold", color=PAL["text"], pad=8)
        ax.set_facecolor(PAL["light"])
        ax.grid(True, axis="x", color=PAL["grid"], lw=0.5)
        _spine(ax)
        for bar, val, n in zip(bars, vals, names):
            is_best = (val == best)
            # Fixed: use actual value formatting, not k-suffix (values are $28, not $28k)
            if key == "val_mae":   fmt = f"${val:.1f}/policy"
            elif key == "val_rmse": fmt = f"${val:.1f}/policy"
            elif key == "val_r2":   fmt = f"{val:.4f}"
            else:                   fmt = f"{val:.1f}%"
            ax.text(bar.get_width() + max(vals) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    fmt + (" ★" if is_best else ""),
                    va="center", fontsize=9,
                    color=PAL["teal"] if is_best else PAL["text"],
                    fontweight="bold" if is_best else "normal")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlim(0, max(vals) * 1.38)

    plt.tight_layout()
    plt.savefig(out_path / "fig2_model_comparison.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved fig2_model_comparison.png")


def fig3_feature_importance(results, best_name, out_path):
    res = results[best_name]
    imp = (
        pd.DataFrame({"Feature": res["feats"],
                      "Importance": res["model"].feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(20)
        .sort_values("Importance")
    )

    def fc(f):
        if f in ("quantum_risk_prob", "q_x_HO", "q_x_RT", "q_excess"):
            return PAL["coral"]
        if f == "Org_Risk_Lag1":
            return PAL["blue"]
        if f.startswith("Category_"):
            return PAL["gold"]
        if f in ("total_population", "median_income",
                 "housing_value", "housing_vacancy_number"):
            return PAL["teal"]
        return PAL["slate"]

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")
    bars = ax.barh(imp["Feature"], imp["Importance"],
                   color=[fc(f) for f in imp["Feature"]],
                   edgecolor="white", linewidth=0.6, height=0.72)

    for bar, feat, val in zip(bars, imp["Feature"], imp["Importance"]):
        if feat in ("quantum_risk_prob", "q_x_HO", "q_x_RT", "q_excess"):
            tag = "  <- QUANTUM"
        elif feat == "Org_Risk_Lag1":
            tag = "  <- ORGANIZER"
        else:
            tag = ""
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}{tag}", va="center", fontsize=8.5,
                color=fc(feat) if tag else PAL["text"],
                fontweight="bold" if tag else "normal")

    ax.set_title(f"Feature Importance — {best_name}\n(GBM, top 20 features)",
                 fontsize=13, fontweight="bold", color=PAL["text"], pad=12)
    ax.set_xlabel("Relative Importance (GBM split gain)", fontsize=10, color=PAL["text"])
    ax.set_facecolor(PAL["light"])
    ax.grid(True, axis="x", color=PAL["grid"], lw=0.5)
    ax.set_xlim(0, imp["Importance"].max() * 1.22)
    _spine(ax)
    ax.legend(handles=[
        Patch(facecolor=PAL["coral"], label="Quantum risk features (Task 1)"),
        Patch(facecolor=PAL["blue"],  label="Organizer risk score (lagged)"),
        Patch(facecolor=PAL["gold"],  label="Policy category"),
        Patch(facecolor=PAL["teal"],  label="Census / geography"),
        Patch(facecolor=PAL["slate"], label="Actuarial / temporal"),
    ], loc="lower right", fontsize=9, framealpha=0.92, edgecolor=PAL["grid"])

    plt.tight_layout()
    plt.savefig(out_path / "fig3_feature_importance.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved fig3_feature_importance.png")


def fig4_risk_comparison(results, out_path):
    """
    Quintile box plots: premium stratification by each risk signal.
    y-axis clipped at p92 so boxes are readable despite outliers.
    """
    res  = results["C — Hybrid Synergy"]
    test = res["test_df"].copy()
    test["Predicted_Rate"] = np.expm1(res["pred_log_te"]).clip(0)
    test["Predicted_Earned_Premium"] = (
        test["Predicted_Rate"] * test["Earned Exposure"].clip(lower=0)
    ).clip(lower=0)

    zip_agg = (
        test.groupby("ZIP")
            .agg(
                Predicted_Earned_Premium=("Predicted_Earned_Premium", "sum"),
                quantum_risk_prob=("quantum_risk_prob", "first"),
                Avg_Org_Fire_Risk_Score=("Avg Fire Risk Score", "mean"),
            )
            .reset_index()
    )

    zip_agg["org_quintile"] = pd.qcut(
        zip_agg["Avg_Org_Fire_Risk_Score"], q=5, duplicates="drop",
        labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])
    zip_agg["q_quintile"] = pd.qcut(
        zip_agg["quantum_risk_prob"], q=5, duplicates="drop",
        labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    fig.suptitle(
        "Complementary Risk Signals — Premium Stratification by Risk Quintile\n"
        "Both signals independently separate low- from high-premium ZIP codes",
        fontsize=12, fontweight="bold", color=PAL["text"],
    )

    gradient_org = ["#A8D5C2", "#6FBFA8", PAL["gold"], "#E08050", PAL["coral"]]
    gradient_q   = ["#A8C4D5", "#6B9FC0", PAL["gold"], "#D07040", PAL["coral"]]
    qlabels      = ["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"]
    y_cap        = zip_agg["Predicted_Earned_Premium"].quantile(0.92)

    for ax, (quintile_col, grads, title, color) in zip(axes, [
        ("org_quintile", gradient_org,
         "Organizer 'Avg Fire Risk Score'\n(per-category, varies within ZIP)", PAL["blue"]),
        ("q_quintile",   gradient_q,
         "Quantum ML Wildfire Probability\n(Task 1 hybrid model, ZIP-level)", PAL["coral"]),
    ]):
        groups = zip_agg.groupby(quintile_col, observed=True)
        data   = [groups.get_group(q)["Predicted_Earned_Premium"].values
                  for q in qlabels if q in groups.groups]
        actual_labels = [q for q in qlabels if q in groups.groups]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", lw=2.5),
                        whiskerprops=dict(color=PAL["slate"], lw=1.2),
                        capprops=dict(color=PAL["slate"], lw=1.2),
                        flierprops=dict(marker=".", color=PAL["slate"], alpha=0.2, ms=3))
        for patch, c in zip(bp["boxes"], grads[:len(data)]):
            patch.set_facecolor(c); patch.set_alpha(0.88)

        means = [np.mean(d) for d in data]
        ax.scatter(range(1, len(means)+1), means, zorder=5,
                   color=color, s=70, marker="D", label="Mean premium", clip_on=True)
        for i, mn in enumerate(means):
            ax.text(i+1, min(mn*1.08, y_cap*0.93),
                    f"${mn/1e6:.1f}M", ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

        ax.set_ylim(0, y_cap)
        ax.set_xticklabels(actual_labels, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=8)
        ax.set_ylabel("Predicted Earned Premium ($)", fontsize=9, color=PAL["text"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
        ax.set_facecolor(PAL["light"])
        ax.grid(True, axis="y", color=PAL["grid"], lw=0.5)
        ax.text(0.97, 0.97,
                f"y-axis clipped at ${y_cap/1e6:.0f}M\n(outliers hidden for clarity)",
                transform=ax.transAxes, fontsize=7.5, color=PAL["slate"],
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.7, edgecolor=PAL["grid"]))
        ax.legend(fontsize=9, loc="upper left")
        _spine(ax)

    plt.tight_layout()
    plt.savefig(out_path / "fig4_risk_signal_comparison.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved fig4_risk_signal_comparison.png")


def fig5_premium_distribution(zip_agg, out_path):
    premiums = zip_agg["Predicted_Earned_Premium"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    fig.suptitle("2021 Predicted Earned Premium — Distribution Across ZIP Codes",
                 fontsize=13, fontweight="bold", color=PAL["text"])

    ax = axes[0]
    cap = premiums.quantile(0.98)
    ax.hist(premiums.clip(upper=cap), bins=60, color=PAL["blue"],
            edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axvline(premiums.median(), color=PAL["coral"], lw=2, ls="--",
               label=f"Median ${premiums.median()/1e6:.2f}M")
    ax.axvline(premiums.mean(),   color=PAL["gold"],  lw=2, ls=":",
               label=f"Mean   ${premiums.mean()/1e6:.2f}M")
    ax.set(xlabel="Predicted Total Premium per ZIP ($)",
           ylabel="Number of ZIP Codes", title="Histogram (clipped at p98)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
    ax.legend(fontsize=9, framealpha=0.92)
    ax.set_facecolor(PAL["light"])
    ax.grid(True, axis="y", color=PAL["grid"], lw=0.5)
    _spine(ax)

    ax2 = axes[1]
    sp  = np.sort(premiums)
    cdf = np.arange(1, len(sp) + 1) / len(sp)
    ax2.plot(sp, cdf, color=PAL["teal"], lw=2.5)
    ax2.fill_betweenx(cdf, sp, alpha=0.15, color=PAL["teal"])
    for pct, ls, col in [(0.5, "--", PAL["coral"]),
                          (0.9, ":",  PAL["gold"]),
                          (0.95, "-.", PAL["slate"])]:
        v = float(np.quantile(sp, pct))
        ax2.axvline(v, color=col, lw=1.4, ls=ls,
                    label=f"p{int(pct*100)}: ${v/1e6:.1f}M")
    ax2.set(xlabel="Predicted Total Premium per ZIP ($)",
            ylabel="Cumulative Fraction of ZIPs",
            title="Cumulative Distribution (CDF)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
    ax2.legend(fontsize=9, framealpha=0.92)
    ax2.set_facecolor(PAL["light"])
    ax2.grid(True, color=PAL["grid"], lw=0.5)
    _spine(ax2)

    plt.tight_layout()
    plt.savefig(out_path / "fig5_premium_distribution.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved fig5_premium_distribution.png")



def fig6_quantum_unique_signal(results, out_path):
    """
    Left:  Rank-rank scatter of org vs quantum risk, coloured by log(premium).
    Right: Clean table of ZIPs where quantum spots high risk org misses,
           filtered to active markets (>$100k premium).
    Uses GridSpec + explicit bbox for table so title never overlaps content.
    """
    import matplotlib.gridspec as gridspec

    res  = results["C — Hybrid Synergy"]
    test = res["test_df"].copy()
    test["Predicted_Rate"] = np.expm1(res["pred_log_te"]).clip(0)
    test["Predicted_Earned_Premium"] = (
        test["Predicted_Rate"] * test["Earned Exposure"].clip(lower=0)
    ).clip(lower=0)

    zip_agg = (
        test.groupby("ZIP")
            .agg(
                Predicted_Earned_Premium=("Predicted_Earned_Premium", "sum"),
                quantum_risk_prob=("quantum_risk_prob", "first"),
                Avg_Org_Fire_Risk_Score=("Avg Fire Risk Score", "mean"),
            )
            .reset_index()
    )

    zip_agg["org_norm"]  = zip_agg["Avg_Org_Fire_Risk_Score"] / (
        zip_agg["Avg_Org_Fire_Risk_Score"].max() + 1e-9)
    zip_agg["q_rank"]    = zip_agg["quantum_risk_prob"].rank(pct=True)
    zip_agg["org_rank"]  = zip_agg["org_norm"].rank(pct=True)
    zip_agg["rank_diff"] = zip_agg["q_rank"] - zip_agg["org_rank"]

    # Table: active markets only
    sig    = zip_agg[zip_agg["Predicted_Earned_Premium"] > 1e5].copy()
    hidden = (sig.nlargest(15, "rank_diff")
              [["ZIP", "quantum_risk_prob", "Avg_Org_Fire_Risk_Score",
                "Predicted_Earned_Premium"]]
              .copy())
    hidden["Quantum Risk"] = (hidden["quantum_risk_prob"] * 100).round(1).astype(str) + "%"
    hidden["Org Score"]    = hidden["Avg_Org_Fire_Risk_Score"].round(2)
    hidden["2021 Premium"] = (hidden["Predicted_Earned_Premium"] / 1e6).map(
        lambda v: f"${v:.2f}M")
    hidden["ZIP"] = hidden["ZIP"].astype(int).astype(str)
    hidden = hidden[["ZIP", "Quantum Risk", "Org Score", "2021 Premium"]].reset_index(drop=True)

    spearman_r = zip_agg["org_rank"].corr(zip_agg["q_rank"])

    # ── Figure layout ─────────────────────────────────────────────────────
    # Extra height + generous top margin so two-line header has room
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.06, right=0.97,
                            top=0.82, bottom=0.08,   # top=0.82 gives 18% for header
                            wspace=0.12)

    # Main title — high up
    fig.text(0.5, 0.97,
             "Where the Quantum Signal Differs from the Organizer Score",
             ha="center", va="top", fontsize=15,
             fontweight="bold", color=PAL["text"])
    # Subtitle — well separated below
    fig.text(0.5, 0.905,
             "Complementary risk detection: each signal flags ZIPs the other misses",
             ha="center", va="top", fontsize=11, color=PAL["slate"])

    # ── LEFT: rank-rank scatter ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(zip_agg["org_rank"], zip_agg["q_rank"],
                    c=np.log1p(zip_agg["Predicted_Earned_Premium"]),
                    cmap="YlOrRd", alpha=0.55, s=20, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("log( Predicted Premium $ )", fontsize=8, color=PAL["text"])
    cb.ax.tick_params(labelsize=8)

    ax.plot([0, 1], [0, 1], color=PAL["slate"], lw=1.5, ls="--",
            label="Perfect agreement", zorder=4)

    ax.text(0.03, 0.97, "Quantum spots\nrisk org misses",
            transform=ax.transAxes, fontsize=9, color=PAL["coral"],
            fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.88, edgecolor=PAL["coral"], lw=1.2))
    ax.text(0.97, 0.03, "Org spots\nrisk quantum\nmisses",
            transform=ax.transAxes, fontsize=9, color=PAL["blue"],
            fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.88, edgecolor=PAL["blue"], lw=1.2))
    ax.text(0.03, 0.03,
            f"Spearman r = {spearman_r:.3f}\n(low = complementary signals)",
            transform=ax.transAxes, fontsize=9, color="white",
            fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=PAL["blue"], alpha=0.88))

    ax.set_xlabel("Organizer Risk Score — Percentile Rank",
                  fontsize=9.5, color=PAL["text"], labelpad=6)
    ax.set_ylabel("Quantum ML Score — Percentile Rank",
                  fontsize=9.5, color=PAL["text"], labelpad=6)
    ax.set_title("Risk Rank Agreement Across 2,118 ZIP Codes\n"
                 "Off-diagonal = unique signal not captured by the other",
                 fontsize=10, fontweight="bold", color=PAL["text"], pad=10)
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_facecolor(PAL["light"])
    ax.grid(True, color=PAL["grid"], lw=0.5, zorder=0)
    _spine(ax)

    # ── RIGHT: table with explicit bbox — title never touches cells ───────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    # Title drawn manually so we fully control its y position
    ax2.text(0.5, 0.99,
             "Top 15 ZIPs: Quantum Flags High Wildfire Risk",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=11, fontweight="bold", color=PAL["coral"])
    ax2.text(0.5, 0.93,
             "Where the Organizer Score Is Low",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=10, fontweight="bold", color=PAL["coral"])
    ax2.text(0.5, 0.875,
             "(filtered to active markets > $100k premium)",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=9, color=PAL["slate"])

    col_labels = ["ZIP\nCode", "Quantum\nRisk %", "Org\nScore", "2021\nPremium"]
    tbl = ax2.table(
        cellText=hidden.values.tolist(),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.82],   # bottom 82% → clear 18% for 3-line header
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    n_cols = len(col_labels)
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor(PAL["coral"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_height(0.085)

    for i in range(1, len(hidden) + 1):
        row_bg = PAL["light"] if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = tbl[(i, j)]
            cell.set_facecolor(row_bg)
            cell.set_height(0.062)
            if j == 1:
                cell.set_text_props(color=PAL["coral"], fontweight="bold")

    for cell in tbl.get_celld().values():
        cell.set_edgecolor(PAL["grid"]); cell.set_linewidth(0.6)

    plt.savefig(out_path / "fig6_quantum_unique_signal.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved fig6_quantum_unique_signal.png")


# =============================================================================
# 7. MAIN
# =============================================================================

def run(data_path: str, risk_path: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("TASK 2 - INSURANCE PREMIUM FORECASTER v2")
    log.info("A: Organizer Risk  |  B: Quantum (Task 1)  |  C: Hybrid")
    log.info("=" * 65)

    df = load_data(data_path, risk_path)
    df, cat_cols, census_cols, q_inter_cols = engineer(df)
    feature_sets = build_feature_sets(df, cat_cols, census_cols, q_inter_cols)
    results = train_all(df, feature_sets)

    best_name = min(results, key=lambda n: results[n]["val_mae"])
    log.info(f"\n  Winner: {best_name}")

    log.info("\n--- Generating visualisations ---")
    fig1_actual_vs_predicted(results, out_path)
    fig2_model_comparison(results, out_path)
    fig3_feature_importance(results, best_name, out_path)
    fig4_risk_comparison(results, out_path)
    fig6_quantum_unique_signal(results, out_path)

    log.info("\n--- Saving prediction CSVs ---")
    _, zip_agg = save_outputs(results, best_name, out_path)
    fig5_premium_distribution(zip_agg, out_path)

    log.info("\n" + "=" * 65)
    log.info("FINAL SUMMARY")
    log.info(f"  {'Model':<32} {'MAE':>9} {'R2':>8} {'MAPE':>8}")
    log.info("  " + "-" * 60)
    for name, res in results.items():
        tag = "  <- WINNER" if name == best_name else ""
        log.info(f"  {name:<32} {res['val_mae']:>9,.1f} "
                 f"{res['val_r2']:>8.4f} {res['val_mape']:>7.1f}%{tag}")
    log.info("=" * 65)
    log.info(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    # Paths resolved relative to this file's location.
    # Expected project layout:
    #   Quantum-Sustainability-Challenge/
    #     data/raw/cal_insurance_fire_census_weather.csv
    #     results/wildfire_risk_scores_2021.csv
    #     src/premium_forecaster_v2.py   <- this file
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_PATH = PROJECT_ROOT / "data"    / "raw"     / "cal_insurance_fire_census_weather.csv"
    RISK_PATH = PROJECT_ROOT / "results"             / "wildfire_risk_scores_2021.csv"
    OUT_PATH  = PROJECT_ROOT / "results" / "task2"

    run(str(DATA_PATH), str(RISK_PATH), str(OUT_PATH))