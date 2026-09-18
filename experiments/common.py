"""Leak-proof helpers for LOOY Eurovision experiments.

Protocol per fold: all transforms (imputation, label encoding, PCA, scaling)
fit on TRAIN rows only, transform TEST rows. Metrics are always report on
held-out predictions, never training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GRAND_FINAL_CSV = "paper-data/Eurovision_Grand_Final_Data_2008_2024.csv"
CANTOBEL_CSV = "song_cantobels.csv"

N_FINALISTS = 25  # grand final has 25-26 entries; scaling uses /24


def load_grand_final() -> pd.DataFrame:
    df = pd.read_csv(GRAND_FINAL_CSV)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def load_merged() -> pd.DataFrame:
    c = pd.read_csv(CANTOBEL_CSV)
    g = load_grand_final()
    return c.merge(g, left_on=["year", "country"], right_on=["year", "to_country"], how="inner")


PCA_SUBSETS = {
    "barkbands": ["barkbands_crest", "barkbands_flatness_db", "barkbands_kurtosis",
                  "barkbands_skewness", "barkbands_spread"],
    "erbbands": ["erbbands_crest", "erbbands_flatness_db", "erbbands_kurtosis",
                 "erbbands_skewness", "erbbands_spread"],
    "melbands": ["melbands_crest", "melbands_flatness_db", "melbands_kurtosis",
                 "melbands_skewness", "melbands_spread"],
    "spectral": ["spectral_centroid", "spectral_complexity", "spectral_decrease",
                 "spectral_energy", "spectral_energyband_high", "spectral_energyband_low",
                 "spectral_energyband_middle_high", "spectral_energyband_middle_low",
                 "spectral_entropy", "spectral_flux", "spectral_kurtosis", "spectral_rms",
                 "spectral_rolloff", "spectral_skewness", "spectral_spread", "spectral_strongpeak"],
}

AUDIO_SCALAR = [
    "length(s)", "bpm", "beats_count", "beats_loudness", "danceability", "onset_rate",
    "chords_changes_rate", "chords_number_rate", "chords_strength", "hpcp_crest",
    "hpcp_entropy", "tuning_diatonic_strength", "tuning_equal_tempered_deviation",
    "tuning_frequency", "tuning_nontempered_energy_ratio", "key_edma_strength",
    "key_krumhansl_strength", "key_temperley_strength", "average_loudness", "dissonance",
    "dynamic_complexity", "hfc", "loudness_ebu128_int", "loudness_ebu128_range",
    "pitch_salience", "zerocrossingrate",
]

AUDIO_CAT = [
    "chords_key", "chords_scale", "key_edma_key", "key_edma_scale",
    "key_krumhansl_key", "key_krumhansl_scale", "key_temperley_key", "key_temperley_scale",
]

LYRICS = [
    "lyrics_english", "lyrics_english_mix", "lyrics_language",
    "total_words", "unique_words", "type_token_ratio",
    "compression_size_reduction", "n_gram_repetitiveness",
]

YOUTUBE = ["total_views", "days_since_upload", "yt_views_per_day"]

RUNNING_ORDER = ["running_final", "running_sf"]

COUNTRY_VOTING = ["LY_SF_reciprocation", "LY_SF_vote", "LY_final_reciprocation", "LY_final_vote"]

COUNTRY = ["to_country"]

# feature sets mirroring the paper's approaches
FULL_FEATURES = list(AUDIO_SCALAR + AUDIO_CAT + LYRICS + YOUTUBE + RUNNING_ORDER + COUNTRY_VOTING + COUNTRY)
NO_YOUTUBE_FEATURES = list(AUDIO_SCALAR + AUDIO_CAT + LYRICS + RUNNING_ORDER + COUNTRY_VOTING + COUNTRY)
# approach 2 in the paper: intrinsic song characteristics + public appeal (no contest data)
INTRINSIC_YOUTUBE = list(AUDIO_SCALAR + AUDIO_CAT + LYRICS + YOUTUBE)
# intrinsic only, no post-contest-exposure signal (honest forward-looking variant)
INTRINSIC_ONLY = list(AUDIO_SCALAR + AUDIO_CAT + LYRICS)


def looy_folds(df: pd.DataFrame, year_col: str = "year"):
    years = np.sort(df[year_col].unique())
    folds = []
    for y in years:
        train = df[df[year_col] != y].reset_index(drop=True)
        test = df[df[year_col] == y].reset_index(drop=True)
        folds.append((y, train, test))
    return folds


def _fit_pca(train: pd.DataFrame, test: pd.DataFrame):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    for name, cols in PCA_SUBSETS.items():
        avail_tr = [c for c in cols if c in train.columns]
        if not avail_tr:
            continue
        scaler = StandardScaler()
        pca = PCA()
        s_tr = scaler.fit_transform(train[avail_tr])
        pca.fit(s_tr)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n = int(np.argmax(cum >= 0.95)) + 1
        tr_pc = pd.DataFrame(pca.transform(scaler.transform(train[avail_tr]))[:, :n],
                             index=train.index, columns=[f"{name}_PC{j+1}" for j in range(n)])
        te_pc = pd.DataFrame(pca.transform(scaler.transform(test[avail_tr]))[:, :n],
                             index=test.index, columns=[f"{name}_PC{j+1}" for j in range(n)])
        train = train.drop(columns=avail_tr).join(tr_pc)
        test = test.drop(columns=avail_tr).join(te_pc)
    return train, test


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    test_df: pd.DataFrame | None = None,
    use_pca: bool = True,
):
    """Return (X_train, y_train, X_test, y_test, feature_names).

    All preprocessing (median imputation, label encoding, PCA, scaling) is fit
    on the train frame and applied to test. PCA subset columns replace the raw
    band/spectral features when use_pca is True.
    """
    train_df = df if test_df is None else df
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else df.copy()

    pca_input = [c for s in PCA_SUBSETS.values() for c in s]
    keep = [c for c in feature_cols if c in train.columns and c != target]
    is_num = [pd.api.types.is_numeric_dtype(train[c]) for c in keep]
    keep_cat = [c for c, n in zip(keep, is_num) if not n]
    keep_num = [c for c, n in zip(keep, is_num) if n]
    pca_cols = [c for c in pca_input if c in train.columns]

    cols_keep = keep + ([c for c in pca_cols if c not in keep])
    train = train[cols_keep + [target]].copy()
    test = test[cols_keep + [target]].copy()

    for col in keep_cat:
        cats = train[col].astype(str).unique()
        mapping = {v: i for i, v in enumerate(sorted(map(str, cats)))}
        train[col] = train[col].astype(str).map(mapping)
        test[col] = test[col].astype(str).map(mapping).fillna(-1)

    for col in keep_num:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    if use_pca:
        train, test = _fit_pca(train, test)

    train = train.infer_objects(copy=False)
    test = test.infer_objects(copy=False)

    ytr = train.pop(target).astype(float).values
    yte = test.pop(target).astype(float).values
    feature_names = list(train.columns)
    return train.values.astype(float), ytr, test.values.astype(float), yte, feature_names


# Generic regularized XGBoost configuration chosen for stability on small folds.
XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0, reg_alpha=1.0,
    min_child_weight=3, eval_metric="rmse",
)


def run_looy_xgb(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    n_seeds: int = 3,
    metrics: tuple[str, ...] = ("r2", "mae", "rmse", "spearman", "avg_rank_error"),
) -> tuple[pd.DataFrame, dict]:
    """Hold-one-year-out XGBoost with per-seed averaging. Held-out metrics only."""
    from xgboost import XGBRegressor
    import pandas as pd

    rows, all_true, all_pred = [], [], []
    for year, train, test in looy_folds(df):
        Xtr, ytr, Xte, yte, names = build_feature_matrix(train, features, target, test_df=test)
        preds = []
        for s in range(n_seeds):
            n = len(Xtr)
            rng = np.random.default_rng(s)
            idx = rng.permutation(n)
            n_val = max(10, n // 5)
            vi, ti = idx[:n_val], idx[n_val:]
            m = XGBRegressor(**{**XGB_PARAMS, "random_state": s, "verbosity": 0})
            m.fit(Xtr[ti], ytr[ti], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
            preds.append(m.predict(Xte))
        yp = np.mean(preds, axis=0)
        m = evaluate(yte, yp)
        m["year"] = int(year)
        rows.append(m)
        all_true.append(yte)
        all_pred.append(yp)

    res = pd.DataFrame(rows)
    pooled = {f"pooled_{k}": round(float(metric(np.concatenate(all_true), np.concatenate(all_pred))), 4)
              for k, metric in [("r2", lambda *a: __import__("sklearn.metrics", fromlist=["r2_score"]).r2_score(*a)),
                                ("mae", lambda *a: __import__("sklearn.metrics", fromlist=["mean_absolute_error"]).mean_absolute_error(*a)),
                                ("rmse", lambda *a: float(np.sqrt(__import__("sklearn.metrics", fromlist=["mean_squared_error"]).mean_squared_error(*a))))]}
    summary = {
        "n_years": int(len(res)),
        "n_seeds": n_seeds,
        "n_features": len(features),
        "mean_r2": round(float(res.r2.mean()), 4),
        "median_r2": round(float(res.r2.median()), 4),
        "pooled_r2": pooled["pooled_r2"],
        "mean_mae": round(float(res.mae.mean()), 4),
        "mean_rmse": round(float(res.rmse.mean()), 4),
        "mean_spearman": round(float(res.spearman.mean()), 4),
        "mean_avg_rank_error": round(float(res.avg_rank_error.mean()), 4),
        "median_avg_rank_error": round(float(res.avg_rank_error.median()), 4),
        "best_year": (int(res.loc[res.r2.idxmax(), "year"]), round(float(res.r2.max()), 3)),
        "worst_year": (int(res.loc[res.r2.idxmin(), "year"]), round(float(res.r2.min()), 3)),
    }
    if "spearman" in metrics:
        summary["pooled_spearman"] = round(float(spearmanr(np.concatenate(all_true),
                                                            np.concatenate(all_pred)).statistic), 4)
    return res, summary


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(yt, yp))
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    if len(yt) >= 3 and np.std(yt) > 0 and np.std(yp) > 0:
        spr = float(spearmanr(yt, yp).statistic)
    else:
        spr = float("nan")
    return {"r2": r2, "mae": mae, "rmse": rmse, "spearman": spr,
            "avg_rank_error": avg_rank_error(yt, yp)}


def avg_rank_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average |predicted_rank - true_rank| within the year (scaled rank 1..25)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    true_rank = np.argsort(np.argsort(yt)) + 1
    pred_rank = np.argsort(np.argsort(yp)) + 1
    return float(np.mean(np.abs(true_rank - pred_rank)))