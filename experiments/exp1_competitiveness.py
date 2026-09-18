"""Experiment 1: predict Rasch song competitiveness (cantobel) from multimodal
features, hold-one-year-out. Baseline to beat: audio embeddings explain ~7% of
variance (Burgoyne et al., 2023).

Run: uv run experiments/exp1_competitiveness.py
"""

from __future__ import annotations

import json
import time

from common import (
    FULL_FEATURES, NO_YOUTUBE_FEATURES, load_merged, looy_folds, run_looy_xgb,
)

BASELINE = 7.2  # % variance explained by audio embeddings in the literature


def main():
    df = load_merged()
    print(f"merged rows: {len(df)}, years: {len(looy_folds(df))}")
    out = {}
    for label, feats in [
        ("cantobel_full_features", FULL_FEATURES),
        ("cantobel_no_youtube", NO_YOUTUBE_FEATURES),
    ]:
        t0 = time.time()
        res, summary = run_looy_xgb(df, feats, "cantobel_median", n_seeds=5)
        print(f"\n### {label}  [{time.time()-t0:.0f}s]")
        print(res[["year", "r2", "mae", "rmse", "spearman", "avg_rank_error"]].round(3).to_string(index=False))
        print(json.dumps(summary))
        print(f"  vs literature baseline {BASELINE}% -> delta {round(summary['median_r2']*100 - BASELINE, 2)}pp (median r2)")
        out[label] = summary
    with open("experiments/results.log", "a") as f:
        f.write(f"\n== exp1 ({time.strftime('%Y-%m-%d %H:%M')}) ==\n")
        f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()