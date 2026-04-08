#!/usr/bin/env python
"""
OOD Experiment 10: Calibration & Uncertainty Quantification

Sub-experiments:
  10a: Calibration curve — bin LOO-target predictions into deciles,
       compare predicted vs actual sensitivity. Report ECE.
  10b: Quantile regression — HGB with loss='quantile' at alpha=0.1
       and 0.9 for 80% prediction intervals. Report coverage and width.
  10c: Novelty detection — isolation forest on training features,
       report NDCG@3 separately for in-distribution vs OOD positions.

Usage:
    python scripts/ood/calibration_uncertainty.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

sys.stdout.reconfigure(line_buffering=True)

# ── Load data ────────────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path("evolve/eval_data/position_data.npz")
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
FEAT_NAMES = list(_data["feature_names"])
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, {len(TARGETS)} targets")

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


# ── Evaluation functions ─────────────────────────────────────────────────────

def ndcg_at_k(y_true, y_score, k=3):
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    discounts = np.log2(np.arange(2, k + 2))
    pred_order = np.argsort(-y_score)[:k]
    dcg = np.sum(y_true[pred_order] / discounts)
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_order] / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def eval_metrics_for_target(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(float(np.argmax(scores[mask]) == np.argmax(y[mask])))
    if len(y) > 5:
        rho, _ = stats.spearmanr(scores, y)
    else:
        rho = 0.0
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "n_groups": len(ndcgs),
    }


# ── 10a: Calibration ────────────────────────────────────────────────────────

def calibration_analysis():
    """Collect LOO-target predictions and compute calibration curve + ECE."""
    print("\n" + "=" * 78)
    print("EXPERIMENT 10A: CALIBRATION CURVE")
    print("=" * 78)

    # Collect out-of-fold predictions for every position
    all_preds = np.full(len(Y), np.nan)
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X[train_mask], Y[train_mask])
        all_preds[lo:hi] = model.predict(X[lo:hi])

    elapsed = time.perf_counter() - t0
    print(f"  LOO-target predictions collected in {elapsed:.0f}s")
    print(f"  Predicted: mean={np.mean(all_preds):.3f}, std={np.std(all_preds):.3f}")
    print(f"  Actual:    mean={np.mean(Y):.3f}, std={np.std(Y):.3f}")

    # Bin into deciles by predicted value
    n_bins = 10
    bin_edges = np.percentile(all_preds, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    bins = []
    for j in range(n_bins):
        mask = (all_preds >= bin_edges[j]) & (all_preds < bin_edges[j + 1])
        if mask.sum() == 0:
            continue
        mean_pred = float(np.mean(all_preds[mask]))
        mean_actual = float(np.mean(Y[mask]))
        bins.append({
            "bin": j + 1,
            "n": int(mask.sum()),
            "mean_predicted": mean_pred,
            "mean_actual": mean_actual,
            "abs_error": abs(mean_pred - mean_actual),
            "pred_range": [float(bin_edges[j]), float(bin_edges[j + 1])],
        })

    # ECE = weighted average of |predicted - actual| per bin
    total_n = sum(b["n"] for b in bins)
    ece = sum(b["abs_error"] * b["n"] / total_n for b in bins)

    # Overall correlation
    r_pearson, _ = stats.pearsonr(all_preds, Y)
    r_spearman, _ = stats.spearmanr(all_preds, Y)
    mae = float(np.mean(np.abs(all_preds - Y)))
    rmse = float(np.sqrt(np.mean((all_preds - Y) ** 2)))

    print(f"\n  Calibration curve (predicted vs actual by decile):")
    print(f"  {'Bin':>4s}  {'N':>7s}  {'Predicted':>10s}  {'Actual':>8s}  {'|Error|':>8s}")
    print("  " + "-" * 45)
    for b in bins:
        print(f"  {b['bin']:>4d}  {b['n']:>7,}  {b['mean_predicted']:>10.3f}  "
              f"{b['mean_actual']:>8.3f}  {b['abs_error']:>8.4f}")

    print(f"\n  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Pearson r: {r_pearson:.4f}")
    print(f"  Spearman rho: {r_spearman:.4f}")

    if ece < 0.05:
        print("  -> WELL CALIBRATED: ECE < 0.05")
    elif ece < 0.10:
        print("  -> MODERATELY CALIBRATED: ECE < 0.10")
    else:
        print("  -> POORLY CALIBRATED: ECE >= 0.10")

    return {
        "bins": bins,
        "ece": float(ece),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(r_pearson),
        "spearman_rho": float(r_spearman),
    }, all_preds


# ── 10b: Quantile regression ────────────────────────────────────────────────

def quantile_regression():
    """HGB quantile regression for 80% prediction intervals."""
    print("\n" + "=" * 78)
    print("EXPERIMENT 10B: QUANTILE REGRESSION (80% PREDICTION INTERVALS)")
    print("=" * 78)

    qr_kwargs_base = {
        "max_iter": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_samples_leaf": 50,
        "random_state": 42,
    }

    all_lo = np.full(len(Y), np.nan)
    all_hi = np.full(len(Y), np.nan)
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False

        model_lo = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.1, **qr_kwargs_base,
        )
        model_hi = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.9, **qr_kwargs_base,
        )

        model_lo.fit(X[train_mask], Y[train_mask])
        model_hi.fit(X[train_mask], Y[train_mask])

        all_lo[lo:hi] = model_lo.predict(X[lo:hi])
        all_hi[lo:hi] = model_hi.predict(X[lo:hi])

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(TARGETS)} targets ({elapsed:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  Quantile regression done in {elapsed:.0f}s")

    # Ensure lo <= hi (quantile regression can occasionally cross)
    all_lo_clean = np.minimum(all_lo, all_hi)
    all_hi_clean = np.maximum(all_lo, all_hi)

    # Coverage: fraction of actual values within [lo, hi]
    covered = (Y >= all_lo_clean) & (Y <= all_hi_clean)
    coverage = float(np.mean(covered))

    # Interval widths
    widths = all_hi_clean - all_lo_clean
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    std_width = float(np.std(widths))

    # Per-target coverage
    target_coverages = []
    for i in range(len(TARGETS)):
        lo_t, hi_t = OFFSETS[i], OFFSETS[i + 1]
        cov = float(np.mean(covered[lo_t:hi_t]))
        width = float(np.mean(widths[lo_t:hi_t]))
        target_coverages.append({
            "target": str(TARGETS[i]),
            "coverage": cov,
            "mean_width": width,
        })

    print(f"\n  Nominal coverage: 80%")
    print(f"  Actual coverage:  {coverage:.1%}")
    print(f"  Mean interval width:   {mean_width:.4f}")
    print(f"  Median interval width: {median_width:.4f}")
    print(f"  Std interval width:    {std_width:.4f}")

    under_cov = [tc for tc in target_coverages if tc["coverage"] < 0.70]
    over_cov = [tc for tc in target_coverages if tc["coverage"] > 0.90]
    print(f"\n  Targets with coverage < 70%: {len(under_cov)}")
    for tc in sorted(under_cov, key=lambda x: x["coverage"])[:5]:
        print(f"    {tc['target']:15s}  coverage={tc['coverage']:.1%}  "
              f"width={tc['mean_width']:.4f}")
    print(f"  Targets with coverage > 90%: {len(over_cov)}")

    if abs(coverage - 0.80) < 0.05:
        print(f"\n  -> WELL CALIBRATED intervals (coverage within 5% of nominal)")
    elif coverage > 0.80:
        print(f"\n  -> CONSERVATIVE intervals (over-covers by {coverage-0.80:.1%})")
    else:
        print(f"\n  -> UNDER-COVERING intervals (under-covers by {0.80-coverage:.1%})")

    return {
        "nominal_coverage": 0.80,
        "actual_coverage": coverage,
        "mean_width": mean_width,
        "median_width": median_width,
        "std_width": std_width,
        "per_target": target_coverages,
    }


# ── 10c: Novelty detection ──────────────────────────────────────────────────

def novelty_detection():
    """Isolation forest for OOD detection — split evaluation by novelty."""
    print("\n" + "=" * 78)
    print("EXPERIMENT 10C: NOVELTY DETECTION (ISOLATION FOREST)")
    print("=" * 78)

    all_ood_scores = np.full(len(Y), np.nan)
    all_preds = np.full(len(Y), np.nan)
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False

        # HGB predictions
        hgb = HistGradientBoostingRegressor(**HGB_KWARGS)
        hgb.fit(X[train_mask], Y[train_mask])
        all_preds[lo:hi] = hgb.predict(X[lo:hi])

        # Isolation forest: positive scores = inlier, negative = outlier
        iso = IsolationForest(
            n_estimators=100, random_state=42, contamination="auto",
        )
        iso.fit(X[train_mask])
        all_ood_scores[lo:hi] = iso.decision_function(X[lo:hi])

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(TARGETS)} targets ({elapsed:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")

    # Split by OOD threshold (decision_function: negative = anomaly)
    threshold = 0.0
    in_dist_mask = all_ood_scores >= threshold
    ood_mask = ~in_dist_mask

    n_in = int(in_dist_mask.sum())
    n_ood = int(ood_mask.sum())
    print(f"\n  In-distribution positions: {n_in:,} ({n_in/len(Y):.1%})")
    print(f"  OOD positions:             {n_ood:,} ({n_ood/len(Y):.1%})")

    # Evaluate NDCG@3 separately for each group, per target
    def eval_subset(pred_vals, mask_subset, label):
        all_ndcgs, all_hits = [], []
        for i in range(len(TARGETS)):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            target_mask = mask_subset[lo:hi]
            if target_mask.sum() < 3:
                continue

            y_sub = Y[lo:hi][target_mask]
            p_sub = pred_vals[lo:hi][target_mask]
            g_sub = GROUPS[lo:hi][target_mask]

            ndcgs, hits = [], []
            for mol_id in np.unique(g_sub):
                mol_mask = g_sub == mol_id
                if mol_mask.sum() < 3:
                    continue
                ndcgs.append(ndcg_at_k(y_sub[mol_mask], p_sub[mol_mask], 3))
                hits.append(float(np.argmax(p_sub[mol_mask]) == np.argmax(y_sub[mol_mask])))
            all_ndcgs.extend(ndcgs)
            all_hits.extend(hits)

        if all_ndcgs:
            result = {
                "label": label,
                "ndcg": float(np.mean(all_ndcgs)),
                "hit_rate": float(np.mean(all_hits)),
                "n_groups": len(all_ndcgs),
            }
            print(f"  {label:40s}  NDCG@3={result['ndcg']:.4f}  "
                  f"Hit@1={result['hit_rate']:.3f}  ({result['n_groups']} groups)")
            return result
        return {"label": label, "ndcg": None, "hit_rate": None, "n_groups": 0}

    print("\n  NDCG@3 by novelty group:")
    in_result = eval_subset(all_preds, in_dist_mask, "In-distribution (HGB)")
    ood_result = eval_subset(all_preds, ood_mask, "Out-of-distribution (HGB)")

    # Also test heuristic on each group
    idx_nh = FEAT_NAMES.index("core_n_heavy")
    heur_scores = -X[:, idx_nh]
    in_heur = eval_subset(heur_scores, in_dist_mask, "In-distribution (heuristic)")
    ood_heur = eval_subset(heur_scores, ood_mask, "Out-of-distribution (heuristic)")

    # OOD score distribution
    print(f"\n  OOD score distribution:")
    print(f"    In-dist mean score:  {np.mean(all_ood_scores[in_dist_mask]):.4f}")
    print(f"    OOD mean score:      {np.mean(all_ood_scores[ood_mask]):.4f}")

    if in_result["ndcg"] is not None and ood_result["ndcg"] is not None:
        gap = in_result["ndcg"] - ood_result["ndcg"]
        print(f"\n  NDCG gap (in-dist - OOD): {gap:+.4f}")
        if abs(gap) < 0.01:
            print("  -> Model performs similarly on novel and familiar features")
        elif gap > 0:
            print("  -> Model degrades on OOD positions (expected)")
        else:
            print("  -> Unusual: model performs BETTER on OOD positions")

    return {
        "threshold": threshold,
        "n_in_dist": n_in,
        "n_ood": n_ood,
        "frac_ood": float(n_ood / len(Y)),
        "in_dist_hgb": in_result,
        "ood_hgb": ood_result,
        "in_dist_heuristic": in_heur,
        "ood_heuristic": ood_heur,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("OOD EXPERIMENT 10: CALIBRATION & UNCERTAINTY QUANTIFICATION")
    print("=" * 78)

    results = {}

    calib_result, _ = calibration_analysis()
    results["calibration"] = calib_result

    results["quantile_regression"] = quantile_regression()

    results["novelty_detection"] = novelty_detection()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("EXPERIMENT 10 SUMMARY")
    print("=" * 78)
    print(f"  Calibration ECE:        {results['calibration']['ece']:.4f}")
    print(f"  80% interval coverage:  {results['quantile_regression']['actual_coverage']:.1%}")
    print(f"  Mean interval width:    {results['quantile_regression']['mean_width']:.4f}")
    print(f"  OOD fraction:           {results['novelty_detection']['frac_ood']:.1%}")

    nd = results["novelty_detection"]
    if nd["in_dist_hgb"]["ndcg"] is not None and nd["ood_hgb"]["ndcg"] is not None:
        print(f"  NDCG in-dist:           {nd['in_dist_hgb']['ndcg']:.4f}")
        print(f"  NDCG OOD:               {nd['ood_hgb']['ndcg']:.4f}")

    # Save
    out_path = Path("outputs/ood/calibration_uncertainty_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
