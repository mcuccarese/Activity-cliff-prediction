#!/usr/bin/env python
"""
OOD Experiment 6: Learning Curve & Low-Data Regime

Tests how model performance degrades as training data decreases.
Answers: "How much data does this approach need?"

Protocol:
  1. Subsample training data to 10%, 25%, 50%, 75%, 100%
  2. LOO-target NDCG@3 at each level
  3. Correlate per-target test size with per-target performance
  4. Simulate rare target: cap training to 50, 100, 200, 500 mols per target

Usage:
    python scripts/ood/learning_curve.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

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


def hit_rate_at_1(y_true, y_score):
    if len(y_true) < 2:
        return 0.0
    return float(np.argmax(y_score) == np.argmax(y_true))


def eval_metrics_for_target(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        n = mask.sum()
        if n < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(hit_rate_at_1(y[mask], scores[mask]))
    if len(y) > 5:
        rho, _ = stats.spearmanr(scores, y)
    else:
        rho = 0.0
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
    }


# ── Learning curve: subsample training data ──────────────────────────────────

def learning_curve_experiment():
    print("\n" + "=" * 78)
    print("EXPERIMENT 6A: LEARNING CURVE (subsample training data)")
    print("=" * 78)

    fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    results = []
    rng = np.random.RandomState(42)

    for frac in fractions:
        t0 = time.perf_counter()
        all_ndcgs, all_hits, all_rhos = [], [], []

        for i in range(len(TARGETS)):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            X_test, y_test = X[lo:hi], Y[lo:hi]
            g_test = GROUPS[lo:hi]

            # Build training set (all other targets)
            train_mask = np.ones(len(Y), dtype=bool)
            train_mask[lo:hi] = False
            X_train_full, y_train_full = X[train_mask], Y[train_mask]

            # Subsample training data
            n_train = len(y_train_full)
            n_sample = max(100, int(n_train * frac))
            if frac < 1.0:
                idx = rng.choice(n_train, size=n_sample, replace=False)
                X_train, y_train = X_train_full[idx], y_train_full[idx]
            else:
                X_train, y_train = X_train_full, y_train_full

            model = HistGradientBoostingRegressor(**HGB_KWARGS)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            m = eval_metrics_for_target(preds, y_test, g_test)
            all_ndcgs.append(m["ndcg"])
            all_hits.append(m["hit_rate"])
            all_rhos.append(m["spearman"])

        elapsed = time.perf_counter() - t0
        result = {
            "fraction": frac,
            "n_train_approx": int(len(Y) * frac * 49 / 50),
            "ndcg": float(np.mean(all_ndcgs)),
            "hit_rate": float(np.mean(all_hits)),
            "spearman": float(np.mean(all_rhos)),
            "ndcg_std": float(np.std(all_ndcgs)),
        }
        results.append(result)
        print(f"  frac={frac:.2f} (~{result['n_train_approx']:,} train)  "
              f"NDCG@3={result['ndcg']:.4f} (+/-{result['ndcg_std']:.4f})  "
              f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}  "
              f"[{elapsed:.0f}s]")

    return results


# ── Per-target size vs performance ───────────────────────────────────────────

def per_target_size_analysis():
    print("\n" + "=" * 78)
    print("EXPERIMENT 6B: PER-TARGET SIZE vs PERFORMANCE")
    print("=" * 78)

    target_sizes = []
    target_ndcgs = []

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        X_test, y_test = X[lo:hi], Y[lo:hi]
        g_test = GROUPS[lo:hi]

        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False
        X_train, y_train = X[train_mask], Y[train_mask]

        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m = eval_metrics_for_target(preds, y_test, g_test)
        target_sizes.append(hi - lo)
        target_ndcgs.append(m["ndcg"])

    target_sizes = np.array(target_sizes)
    target_ndcgs = np.array(target_ndcgs)

    rho, p = stats.spearmanr(target_sizes, target_ndcgs)
    print(f"\n  Correlation between test-set size and NDCG@3:")
    print(f"    Spearman rho = {rho:.4f}, p = {p:.4f}")

    # Show top/bottom 5
    order = np.argsort(target_ndcgs)
    print(f"\n  Bottom 5 targets (worst NDCG@3):")
    for idx in order[:5]:
        print(f"    {str(TARGETS[idx]):20s}  NDCG@3={target_ndcgs[idx]:.4f}  "
              f"n_positions={target_sizes[idx]:,}")
    print(f"\n  Top 5 targets (best NDCG@3):")
    for idx in order[-5:]:
        print(f"    {str(TARGETS[idx]):20s}  NDCG@3={target_ndcgs[idx]:.4f}  "
              f"n_positions={target_sizes[idx]:,}")

    return {
        "size_ndcg_spearman": float(rho),
        "size_ndcg_pvalue": float(p),
        "per_target": [
            {"target": str(TARGETS[i]), "n_positions": int(target_sizes[i]),
             "ndcg": float(target_ndcgs[i])}
            for i in range(len(TARGETS))
        ],
    }


# ── Rare target simulation ──────────────────────────────────────────────────

def rare_target_simulation():
    print("\n" + "=" * 78)
    print("EXPERIMENT 6C: RARE TARGET SIMULATION")
    print("  (cap each training target to N molecules)")
    print("=" * 78)

    caps = [50, 100, 200, 500, 1000, None]
    results = []
    rng = np.random.RandomState(42)

    for cap in caps:
        t0 = time.perf_counter()
        all_ndcgs = []

        for i in range(len(TARGETS)):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            X_test, y_test = X[lo:hi], Y[lo:hi]
            g_test = GROUPS[lo:hi]

            # Build capped training set
            X_train_parts, y_train_parts = [], []
            for j in range(len(TARGETS)):
                if j == i:
                    continue
                jlo, jhi = OFFSETS[j], OFFSETS[j + 1]

                if cap is not None and (jhi - jlo) > cap:
                    idx = rng.choice(jhi - jlo, size=cap, replace=False) + jlo
                    X_train_parts.append(X[idx])
                    y_train_parts.append(Y[idx])
                else:
                    X_train_parts.append(X[jlo:jhi])
                    y_train_parts.append(Y[jlo:jhi])

            X_train = np.vstack(X_train_parts)
            y_train = np.concatenate(y_train_parts)

            model = HistGradientBoostingRegressor(**HGB_KWARGS)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            m = eval_metrics_for_target(preds, y_test, g_test)
            all_ndcgs.append(m["ndcg"])

        elapsed = time.perf_counter() - t0
        cap_label = str(cap) if cap else "unlimited"
        result = {
            "cap_per_target": cap,
            "n_train_approx": int(np.mean([
                min(cap or 1e9, OFFSETS[j+1] - OFFSETS[j])
                for j in range(len(TARGETS))
            ]) * 49),
            "ndcg": float(np.mean(all_ndcgs)),
            "ndcg_std": float(np.std(all_ndcgs)),
        }
        results.append(result)
        print(f"  cap={cap_label:>10s} (~{result['n_train_approx']:,} train)  "
              f"NDCG@3={result['ndcg']:.4f} (+/-{result['ndcg_std']:.4f})  "
              f"[{elapsed:.0f}s]")

    return results


def main():
    print("=" * 78)
    print("OOD EXPERIMENT 6: LEARNING CURVE & LOW-DATA REGIME")
    print("=" * 78)

    results = {}
    results["learning_curve"] = learning_curve_experiment()
    results["per_target_size"] = per_target_size_analysis()
    results["rare_target"] = rare_target_simulation()

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    lc = results["learning_curve"]
    print("\nLearning curve (NDCG@3 vs training fraction):")
    for r in lc:
        bar = "#" * int(r["ndcg"] * 40)
        print(f"  {r['fraction']:>5.0%}  {r['ndcg']:.4f}  {bar}")

    drop_5_to_100 = lc[-1]["ndcg"] - lc[0]["ndcg"]
    print(f"\n  Total drop from 100% to 5%: {drop_5_to_100:+.4f} NDCG@3")
    if abs(drop_5_to_100) < 0.01:
        print("  -> Model is very data-efficient (learns general chemistry rules)")
    elif abs(drop_5_to_100) < 0.03:
        print("  -> Model degrades gracefully with less data")
    else:
        print("  -> Model is data-hungry; performance drops significantly")

    # Save
    out_path = Path("outputs/ood/learning_curve_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
