#!/usr/bin/env python
"""
OOD Experiment 1: Feature Ablation Study

Quantifies the incremental predictive value of each feature group.
Answers the critical question: "Does the model learn anything beyond
'smaller scaffolds have more sensitive positions'?"

Configurations tested:
  A. Random baseline (shuffled scores)
  B. Global mean (constant prediction)
  C. core_n_heavy only — HGB on 1 feature
  D. core_n_heavy + core_n_rings — topology only, HGB (2 feat)
  E. 3D context only — HGB (9 feat, no topology)
  F. Full model — HGB (11 feat)
  G. Full model minus core_n_heavy (10 feat)
  H. Full model minus gasteiger_charge (10 feat)
  I. core_n_heavy heuristic (raw feature, no ML)
  J. -SASA heuristic (raw feature, no ML)

All evaluated with LOO-target NDCG@3, Hit@1, Spearman.

Usage:
    python scripts/ood/ablation_study.py
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
print(f"Features: {FEAT_NAMES}")

# Feature indices
CTX_COLS = [i for i, n in enumerate(FEAT_NAMES) if n.startswith("ctx_")]
TOPO_COLS = [i for i, n in enumerate(FEAT_NAMES) if n.startswith("core_")]
IDX_N_HEAVY = FEAT_NAMES.index("core_n_heavy")
IDX_N_RINGS = FEAT_NAMES.index("core_n_rings")
IDX_CHARGE = FEAT_NAMES.index("ctx_gasteiger_charge")
IDX_SASA = FEAT_NAMES.index("ctx_sasa_attach")


# ── Evaluation functions ─────────────────────────────────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
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


def hit_rate_at_1(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
        "n_groups": len(ndcgs),
    }


# ── Non-ML scoring ──────────────────────────────────────────────────────────

def score_heuristic(score_fn, label):
    scores = score_fn(X)
    all_ndcgs, all_hits, all_rhos = [], [], []
    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        m = eval_metrics_for_target(scores[lo:hi], Y[lo:hi], GROUPS[lo:hi])
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])
    result = {
        "label": label,
        "type": "heuristic",
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
    }
    print(f"  {label:55s}  NDCG@3={result['ndcg']:.4f} (±{result['ndcg_std']:.4f})  "
          f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}")
    return result


# ── LOO-target HGB evaluation ───────────────────────────────────────────────

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


def loo_target_hgb(X_in, label, feat_names=None):
    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()
    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        X_test, y_test = X_in[lo:hi], Y[lo:hi]
        g_test = GROUPS[lo:hi]
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False
        X_train, y_train = X_in[train_mask], Y[train_mask]
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = eval_metrics_for_target(preds, y_test, g_test)
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])
    elapsed = time.perf_counter() - t0
    result = {
        "label": label,
        "type": "hgb",
        "n_features": X_in.shape[1],
        "feature_names": feat_names or [],
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "ndcg_per_target": {str(t): float(v) for t, v in zip(TARGETS, all_ndcgs)},
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  {label:55s}  NDCG@3={result['ndcg']:.4f} (±{result['ndcg_std']:.4f})  "
          f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}  "
          f"[{elapsed:.0f}s]")
    return result


def main():
    print("\n" + "=" * 80)
    print("OOD EXPERIMENT 1: FEATURE ABLATION STUDY")
    print("=" * 80)

    results = []

    # ── Heuristic baselines (no ML) ──────────────────────────────────────
    print("\n--- Heuristic baselines (no ML training) ---")

    results.append(score_heuristic(
        lambda _: np.random.RandomState(42).randn(len(Y)),
        "A. Random",
    ))
    results.append(score_heuristic(
        lambda _: np.full(len(Y), Y.mean()),
        "B. Global mean (constant)",
    ))
    # core_n_heavy has NEGATIVE correlation with sensitivity (larger core -> less sensitive)
    # so use negative sign for ranking
    results.append(score_heuristic(
        lambda x: -x[:, IDX_N_HEAVY],
        "I. -core_n_heavy heuristic (smaller = more sensitive)",
    ))
    results.append(score_heuristic(
        lambda x: -x[:, IDX_SASA],
        "J. -SASA heuristic (less exposed = more sensitive)",
    ))

    # ── HGB with different feature subsets ───────────────────────────────
    print("\n--- HGB (leave-one-target-out, NDCG@3) ---")

    # C. core_n_heavy only
    results.append(loo_target_hgb(
        X[:, [IDX_N_HEAVY]],
        "C. core_n_heavy only (1 feat)",
        ["core_n_heavy"],
    ))

    # D. Topology only
    results.append(loo_target_hgb(
        X[:, TOPO_COLS],
        "D. core_n_heavy + core_n_rings (2 feat)",
        [FEAT_NAMES[i] for i in TOPO_COLS],
    ))

    # E. 3D context only
    results.append(loo_target_hgb(
        X[:, CTX_COLS],
        "E. 3D context only (9 feat, no topology)",
        [FEAT_NAMES[i] for i in CTX_COLS],
    ))

    # F. Full model
    results.append(loo_target_hgb(
        X,
        "F. Full model (11 feat)",
        FEAT_NAMES,
    ))

    # G. Full minus core_n_heavy
    drop_heavy = [i for i in range(X.shape[1]) if i != IDX_N_HEAVY]
    results.append(loo_target_hgb(
        X[:, drop_heavy],
        "G. Full minus core_n_heavy (10 feat)",
        [FEAT_NAMES[i] for i in drop_heavy],
    ))

    # H. Full minus gasteiger_charge
    drop_charge = [i for i in range(X.shape[1]) if i != IDX_CHARGE]
    results.append(loo_target_hgb(
        X[:, drop_charge],
        "H. Full minus gasteiger_charge (10 feat)",
        [FEAT_NAMES[i] for i in drop_charge],
    ))

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Config':<58s} {'NDCG@3':>8s} {'Hit@1':>7s} {'Spearman':>10s}")
    print("-" * 85)
    for r in results:
        print(f"  {r['label']:<56s} {r['ndcg']:.4f}   {r['hit_rate']:.3f}   {r['spearman']:.4f}")

    # Key comparisons
    print("\n--- Key incremental gains ---")
    f_idx = next(i for i, r in enumerate(results) if r["label"].startswith("F."))
    d_idx = next(i for i, r in enumerate(results) if r["label"].startswith("D."))
    e_idx = next(i for i, r in enumerate(results) if r["label"].startswith("E."))
    c_idx = next(i for i, r in enumerate(results) if r["label"].startswith("C."))

    lift_3d = results[f_idx]["ndcg"] - results[d_idx]["ndcg"]
    lift_topo = results[f_idx]["ndcg"] - results[e_idx]["ndcg"]
    lift_ml = results[d_idx]["ndcg"] - results[c_idx]["ndcg"]
    print(f"  3D context lift (F vs D):  {lift_3d:+.4f} NDCG@3")
    print(f"  Topology lift (F vs E):    {lift_topo:+.4f} NDCG@3")
    print(f"  n_rings addition (D vs C): {lift_ml:+.4f} NDCG@3")

    print("\n  INTERPRETATION:")
    if lift_3d > 0.01:
        print(f"  [YES] 3D context features add meaningful signal (+{lift_3d:.4f})")
    elif lift_3d > 0.005:
        print(f"  [MARGINAL] 3D context features add marginal signal (+{lift_3d:.4f})")
    else:
        print(f"  [NO] 3D context features add negligible NDCG signal (+{lift_3d:.4f})")
        print(f"    The model is essentially 'smaller scaffolds = more sensitive'")
        print(f"    BUT check Spearman: 3D features may help overall correlation")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = Path("outputs/ood/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
