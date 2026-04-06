"""
Data preprocessing and feature engineering for wildfire prediction.

Loads raw wildfire + weather data, engineers 27 features per zip code per year,
and creates train/val/test splits for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pathlib import Path


FEATURE_COLS_WEATHER = [
    "avg_tmax", "max_tmax", "min_tmax", "tmax_range",
    "avg_tmin", "max_tmin", "min_tmin",
    "total_precip", "avg_precip", "max_precip", "dry_months", "temp_std",
    "summer_tmax", "summer_tmin", "summer_precip", "summer_dry",
    "fall_tmax", "fall_precip",
    "aridity", "summer_aridity",
]

FEATURE_COLS_HIST = [
    "hist_count", "hist_total_acres", "hist_max_acres",
    "hist_avg_acres", "hist_years", "had_fire_last_year", "fire_frequency",
]

FEATURE_COLS = FEATURE_COLS_WEATHER + FEATURE_COLS_HIST

TRAIN_YEARS = [2019, 2020, 2021]
VAL_YEAR = 2022
TEST_YEAR = 2023
MAX_WEATHER_YEAR = 2021


def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(data_dir) / "wildfire_weather.csv"
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def split_fire_weather(df):
    fire = df[df["OBJECTID"].notna()].copy()
    weather = df[df["OBJECTID"].isna()].copy()
    weather["year"] = weather["year_month"].str[:4].astype(int)
    weather["month"] = weather["year_month"].str[5:7].astype(int)
    fire["year"] = fire["Year"].astype(int)
    print(f"Fire incidents: {len(fire):,}  |  Weather observations: {len(weather):,}")
    return fire, weather


def build_weather_features(weather):
    annual = weather.groupby(["zip", "year"]).agg(
        avg_tmax=("avg_tmax_c", "mean"),
        max_tmax=("avg_tmax_c", "max"),
        min_tmax=("avg_tmax_c", "min"),
        tmax_range=("avg_tmax_c", lambda x: x.max() - x.min()),
        avg_tmin=("avg_tmin_c", "mean"),
        max_tmin=("avg_tmin_c", "max"),
        min_tmin=("avg_tmin_c", "min"),
        total_precip=("tot_prcp_mm", "sum"),
        avg_precip=("tot_prcp_mm", "mean"),
        max_precip=("tot_prcp_mm", "max"),
        dry_months=("tot_prcp_mm", lambda x: (x < 5).sum()),
        temp_std=("avg_tmax_c", "std"),
    ).reset_index()

    summer = weather[weather["month"].between(6, 9)]
    summer_agg = summer.groupby(["zip", "year"]).agg(
        summer_tmax=("avg_tmax_c", "mean"),
        summer_tmin=("avg_tmin_c", "mean"),
        summer_precip=("tot_prcp_mm", "sum"),
        summer_dry=("tot_prcp_mm", lambda x: (x < 5).sum()),
    ).reset_index()

    fall = weather[weather["month"].between(9, 11)]
    fall_agg = fall.groupby(["zip", "year"]).agg(
        fall_tmax=("avg_tmax_c", "mean"),
        fall_precip=("tot_prcp_mm", "sum"),
    ).reset_index()

    features = annual.merge(summer_agg, on=["zip", "year"], how="left")
    features = features.merge(fall_agg, on=["zip", "year"], how="left")
    features["aridity"] = features["avg_tmax"] / (features["total_precip"] + 1)
    features["summer_aridity"] = features["summer_tmax"] / (features["summer_precip"] + 1)

    print(f"Weather features: {features.shape}")
    return features


def build_fire_history(fire, all_zips):
    parts = []
    for yr in range(2019, 2024):
        past = fire[fire["year"] < yr]
        if len(past) == 0:
            continue
        agg = past.groupby("zip").agg(
            hist_count=("OBJECTID", "count"),
            hist_total_acres=("GIS_ACRES", "sum"),
            hist_max_acres=("GIS_ACRES", "max"),
            hist_avg_acres=("GIS_ACRES", "mean"),
            hist_years=("Year", "nunique"),
        ).reset_index()
        recent = set(fire[fire["year"] == yr - 1]["zip"].dropna().unique())
        agg["had_fire_last_year"] = agg["zip"].apply(lambda z: 1 if z in recent else 0)
        agg["fire_frequency"] = agg["hist_count"] / max(1, yr - 2018)
        agg["year"] = yr
        parts.append(agg)

    yr2018 = pd.DataFrame({
        "zip": all_zips, "year": 2018,
        "hist_count": 0, "hist_total_acres": 0, "hist_max_acres": 0,
        "hist_avg_acres": 0, "hist_years": 0, "had_fire_last_year": 0,
        "fire_frequency": 0.0,
    })
    history = pd.concat([yr2018] + parts, ignore_index=True)
    print(f"Fire history: {history.shape}")
    return history


def build_target(fire, all_zips):
    rows = []
    for yr in range(2018, 2024):
        fire_zips = set(fire[fire["year"] == yr]["zip"].dropna().unique())
        for z in all_zips:
            rows.append({"zip": z, "year": yr, "fire_occurred": 1 if z in fire_zips else 0})
    return pd.DataFrame(rows)


def build_dataset(data_dir: str = "data/raw"):
    df = load_raw_data(data_dir)
    fire, weather = split_fire_weather(df)
    all_zips = sorted(weather["zip"].dropna().unique())

    weather_feat = build_weather_features(weather)
    fire_hist = build_fire_history(fire, np.array(all_zips))
    target = build_target(fire, np.array(all_zips))

    wcols = [c for c in weather_feat.columns if c not in ["zip", "year"]]
    hcols = [c for c in fire_hist.columns if c not in ["zip", "year"]]

    parts = []
    for tyr in range(2019, 2024):
        wyr = min(tyr - 1, MAX_WEATHER_YEAR)
        w = weather_feat[weather_feat["year"] == wyr]
        h = fire_hist[fire_hist["year"] == tyr]
        t = target[target["year"] == tyr][["zip", "fire_occurred"]]
        m = t.merge(w[["zip"] + wcols], on="zip", how="left")
        m = m.merge(h[["zip"] + hcols], on="zip", how="left")
        m["target_year"] = tyr
        parts.append(m)

    final = pd.concat(parts, ignore_index=True)
    for c in hcols:
        final[c] = final[c].fillna(0)
    final = final.fillna(0)

    print(f"\nFinal dataset: {final.shape}")
    print(f"Nulls remaining: {final[FEATURE_COLS].isna().sum().sum()}")
    return final


def get_splits(final_df, n_pca: int = 8):
    train = final_df[final_df["target_year"].isin(TRAIN_YEARS)]
    val = final_df[final_df["target_year"] == VAL_YEAR]
    test = final_df[final_df["target_year"] == TEST_YEAR]

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

    print(f"Train: {X_train_s.shape}, pos rate: {y_train.mean():.3f}")
    print(f"Val:   {X_val_s.shape}, pos rate: {y_val.mean():.3f}")
    print(f"Test:  {X_test_s.shape}, pos rate: {y_test.mean():.3f}")
    print(f"PCA {n_pca} components: {pca.explained_variance_ratio_.sum():.1%} variance")

    return {
        "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
        "X_train_pca": X_train_pca, "X_val_pca": X_val_pca, "X_test_pca": X_test_pca,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "pca": pca, "test_zips": test["zip"].values,
    }


def prepare_quantum_data(splits, n_samples: int = 500, seed: int = 42):
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
    X_va = np.clip(scaler_q.transform(splits["X_val_pca"]), 0, np.pi)
    X_te = np.clip(scaler_q.transform(splits["X_test_pca"]), 0, np.pi)

    # Also transform the full training set for hybrid model
    X_tr_full = np.clip(scaler_q.transform(splits["X_train_pca"]), 0, np.pi)

    print(f"Quantum train: {X_tr.shape}, pos rate: {y_sub.mean():.2f}")
    return X_tr, y_sub, X_va, X_te, X_tr_full, scaler_q


if __name__ == "__main__":
    final_df = build_dataset("data/raw")
    out_path = Path("data/processed/wildfire_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    splits = get_splits(final_df)
    print("\nData preprocessing complete.")
