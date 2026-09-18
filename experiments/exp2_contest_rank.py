"""Experiment 2: predict grand-final contest ranking (place_contest_scaled),
hold-one-year-out, matching the Adamska & Reiss (2025) protocol.

Paper grand-final XGBoost reference points:
  - approach 2 (intrinsic + public appeal, INTRINSIC_YOUTUBE): R2 0.30,
    avg rank error 4.96, Spearman 0.57
  - approach 3 (intrinsic + appeal + Eurovision data, FULL_FEATURES): R2 0.09,
    avg rank error 6.80

NOTE: cantobel_median as a feature is EXCLUDED here. It correlates ~-0.89 with
the target (both are derived from the same contest votes), so it would be
leakage, not prediction.

Run: uv run experiments/exp2_contest_rank.py
"""

from __future__ import annotations

import json
import time

from common import (
    FULL_FEATURES, INTRINSIC_ONLY, INTRINSIC_YOUTUBE, NO_YOUTUBE_FEATURES,
    load_grand_final, looy_folds, run_looy_xgb,
)

TARGET = "place_contest_scaled"

PAPER = {
    "intrinsic_youtube": {"r2": 0.30, "avg_rank_error": 4.96, "spearman": 0.57, "note": "paper approach 2"},
    "full_features": {"r2": 0.09, "avg_rank_error": 6.80, "spearman": 0.57, "note": "paper approach 3"},
}


def main():
    df = load_grand_final()
    configs = [
        ("intrinsic_youtube", INTRINSIC_YOUTUBE),
        ("full_features", FULL_FEATURES),
        ("no_youtube", NO_YOUTUBE_FEATURES),
        ("intrinsic_only", INTRINSIC_ONLY),
    ]

    out = {}
    for label, feats in configs:
        t0 = time.time()
        res, summary = run_looy_xgb(df, feats, TARGET, n_seeds=5)
        summary["n_rows"] = int(len(df))
        summary["features"] = label
        print(f"\n### {label}  [{time.time()-t0:.0f}s]")
        print(res[["year", "r2", "mae", "rmse", "spearman", "avg_rank_error"]].round(3).to_string(index=False))
        print(json.dumps(summary))
        if label in PAPER:
            p = PAPER[label]
            print(f"  vs paper {p['note']}: r2 {p['r2']}->{summary['median_r2']}, "
                  f"ARE {p['avg_rank_error']}->{summary['mean_avg_rank_error']}, "
                  f"spearman {p['spearman']}->{summary['mean_spearman']}")
        out[label] = summary

    with open("experiments/results.log", "a") as f:
        f.write(f"\n== exp2 ({time.strftime('%Y-%m-%d %H:%M')}) ==\n")
        f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()