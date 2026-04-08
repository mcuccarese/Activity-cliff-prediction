#!/usr/bin/env python
"""
OOD Experiment 2: Novel Scaffold Holdout

Tests generalization to cores never seen during training.

Protocol:
  A. LOO-target with seen/unseen core breakdown:
     For each held-out target, split positions into:
       - "seen cores": core appears in >= 1 training target
       - "unseen cores": core appears ONLY in the held-out target
     Report NDCG@3 separately for each group.

  B. Aggressive scaffold holdout:
     Randomly hold out 20% of cores across ALL targets.
     Train only on the remaining 80% of cores.
     Test on held-out cores.

Usage:
    python scripts/ood/novel_scaffold_holdout.py
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

# We need core_smiles to identify shared scaffolds
# Reload from the MMP data
print("Loading MMP data to identify core sharing across targets...")
import pandas as pd

mmps = pd.read_parquet(
    "outputs/mmps/all_mmps.parquet",
    columns=["target_chembl_id", "mol_from", "core_smiles"],
)
# Deduplicate to (core, target)
core_target = mmps[["core_smiles", "target_chembl_id"]].drop_duplicates()
del mmps

# Map: core -> set of targets it appears in
from collections import defaultdict
core_to_targets: dict[str, set[str]] = defaultdict(set)
for row in core_target.itertuples():
    core_to_targets[row.core_smiles].add(row.target_chembl_id)

print(f"  {len(core_to_targets):,} unique cores across {len(TARGETS)} targets")

# Count how many targets each core appears in
n_targets_per_core = {c: len(ts) for c, ts in core_to_targets.items()}
shared_cores = sum(1 for v in n_targets_per_core.values() if v > 1)
unique_cores = sum(1 for v in n_targets_per_core.values() if v == 1)
print(f"  Cores in >1 target: {shared_cores:,} ({100*shared_cores/len(core_to_targets):.1f}%)")
print(f"  Cores in 1 target:  {unique_cores:,} ({100*unique_cores/len(core_to_targets):.1f}%)")

# We also need core_smiles per position in the position_data
# Re-derive from the position data preparation
# The position data was sorted by (target, mol_from, core_smiles)
# We need to reload the mapping

print("Loading position-level data with core_smiles...")
mmps_pos = pd.read_parquet(
    "outputs/mmps/all_mmps.parquet",
    columns=["target_chembl_id", "mol_from", "core_smiles", "abs_delta_pActivity"],
)

# Aggregate to position level (same as prepare_position_data.py)
pos = (
    mmps_pos
    .groupby(["core_smiles", "target_chembl_id"])
    .agg(
        sensitivity_mean=("abs_delta_pActivity", "mean"),
        n_mmps=("abs_delta_pActivity", "count"),
    )
    .reset_index()
)
pos = pos[pos["n_mmps"] >= 3].reset_index(drop=True)

# Map mol_from to positions
mol_pos = (
    mmps_pos[["mol_from", "core_smiles", "target_chembl_id"]]
    .drop_duplicates()
)
del mmps_pos

mol_pos = mol_pos.merge(pos[["core_smiles", "target_chembl_id"]], on=["core_smiles", "target_chembl_id"])

# Filter for >= 3 positions per (mol, target)
pos_count = (
    mol_pos.groupby(["mol_from", "target_chembl_id"]).size()
    .reset_index(name="n_pos")
)
pos_count = pos_count[pos_count["n_pos"] >= 3]
mol_pos = mol_pos.merge(
    pos_count[["mol_from", "target_chembl_id"]],
    on=["mol_from", "target_chembl_id"],
)

# Sort to match position_data.npz ordering
mol_pos = mol_pos.sort_values(
    ["target_chembl_id", "mol_from", "core_smiles"]
).reset_index(drop=True)

CORE_SMILES = mol_pos["core_smiles"].values
assert len(CORE_SMILES) == len(Y), f"Mismatch: {len(CORE_SMILES)} vs {len(Y)}"
print(f"  Aligned {len(CORE_SMILES):,} core_smiles with position data")


# ── Evaluation functions ─────────────────────────────────────────────────────

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


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
        "n_groups": len(ndcgs),
    }


# ── Experiment A: LOO-target with seen/unseen core breakdown ─────────────────

def experiment_a():
    print("\n" + "=" * 78)
    print("EXPERIMENT 2A: LOO-TARGET WITH SEEN vs UNSEEN CORE BREAKDOWN")
    print("=" * 78)

    results_seen = []
    results_unseen = []
    per_target = []
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        target = str(TARGETS[i])

        X_test, y_test = X[lo:hi], Y[lo:hi]
        g_test = GROUPS[lo:hi]
        cores_test = CORE_SMILES[lo:hi]

        # Build training set
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False
        X_train, y_train = X[train_mask], Y[train_mask]

        # Train model
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Identify seen vs unseen cores
        train_cores = set(CORE_SMILES[train_mask])

        seen_mask = np.array([c in train_cores for c in cores_test])
        unseen_mask = ~seen_mask

        n_seen = seen_mask.sum()
        n_unseen = unseen_mask.sum()

        # Evaluate on full test, seen, and unseen
        m_all = eval_metrics_for_target(preds, y_test, g_test)

        target_result = {
            "target": target,
            "n_positions": int(hi - lo),
            "n_seen": int(n_seen),
            "n_unseen": int(n_unseen),
            "frac_unseen": float(n_unseen / (hi - lo)) if (hi - lo) > 0 else 0,
            "ndcg_all": m_all["ndcg"],
        }

        if n_seen > 10:
            m_seen = eval_metrics_for_target(preds[seen_mask], y_test[seen_mask],
                                              g_test[seen_mask])
            target_result["ndcg_seen"] = m_seen["ndcg"]
            target_result["spearman_seen"] = m_seen["spearman"]
            results_seen.append(m_seen["ndcg"])

        if n_unseen > 10:
            m_unseen = eval_metrics_for_target(preds[unseen_mask], y_test[unseen_mask],
                                                g_test[unseen_mask])
            target_result["ndcg_unseen"] = m_unseen["ndcg"]
            target_result["spearman_unseen"] = m_unseen["spearman"]
            results_unseen.append(m_unseen["ndcg"])

        per_target.append(target_result)

    elapsed = time.perf_counter() - t0

    # Summary
    print(f"\n  Targets with enough seen positions:   {len(results_seen)}")
    print(f"  Targets with enough unseen positions: {len(results_unseen)}")

    if results_seen:
        print(f"\n  SEEN cores (core in >=1 training target):")
        print(f"    Mean NDCG@3 = {np.mean(results_seen):.4f} +/- {np.std(results_seen):.4f}")
    if results_unseen:
        print(f"\n  UNSEEN cores (core ONLY in test target):")
        print(f"    Mean NDCG@3 = {np.mean(results_unseen):.4f} +/- {np.std(results_unseen):.4f}")

    if results_seen and results_unseen:
        gap = np.mean(results_seen) - np.mean(results_unseen)
        print(f"\n  Gap (seen - unseen): {gap:+.4f} NDCG@3")
        if abs(gap) < 0.01:
            print(f"  -> Minimal gap: model generalizes well to novel scaffolds")
        elif gap > 0.01:
            print(f"  -> Seen cores outperform: some scaffold memorization present")
        else:
            print(f"  -> Unseen cores outperform: surprising, check data")

    print(f"\n  [{elapsed:.0f}s]")

    return {
        "seen_ndcg_mean": float(np.mean(results_seen)) if results_seen else None,
        "seen_ndcg_std": float(np.std(results_seen)) if results_seen else None,
        "unseen_ndcg_mean": float(np.mean(results_unseen)) if results_unseen else None,
        "unseen_ndcg_std": float(np.std(results_unseen)) if results_unseen else None,
        "n_targets_with_seen": len(results_seen),
        "n_targets_with_unseen": len(results_unseen),
        "per_target": per_target,
    }


# ── Experiment B: Scaffold-level holdout ─────────────────────────────────────

def experiment_b():
    print("\n" + "=" * 78)
    print("EXPERIMENT 2B: SCAFFOLD-LEVEL HOLDOUT (20% cores held out)")
    print("=" * 78)

    rng = np.random.RandomState(42)
    all_cores = list(set(CORE_SMILES))
    n_holdout = int(len(all_cores) * 0.2)
    holdout_cores = set(rng.choice(all_cores, size=n_holdout, replace=False))
    print(f"  Total cores: {len(all_cores):,}")
    print(f"  Held out:    {n_holdout:,} (20%)")

    holdout_mask = np.array([c in holdout_cores for c in CORE_SMILES])
    train_mask = ~holdout_mask

    n_train = train_mask.sum()
    n_test = holdout_mask.sum()
    print(f"  Training positions: {n_train:,}")
    print(f"  Test positions:     {n_test:,}")

    X_train, y_train = X[train_mask], Y[train_mask]
    X_test, y_test = X[holdout_mask], Y[holdout_mask]
    g_test = GROUPS[holdout_mask]

    t0 = time.perf_counter()
    model = HistGradientBoostingRegressor(**HGB_KWARGS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    # Evaluate (single global test, no LOO-target)
    m = eval_metrics_for_target(preds, y_test, g_test)
    rho_global, p_global = stats.spearmanr(preds, y_test)

    print(f"\n  NDCG@3 on held-out scaffolds: {m['ndcg']:.4f}")
    print(f"  Hit@1 on held-out scaffolds:  {m['hit_rate']:.3f}")
    print(f"  Spearman (global):            {rho_global:.4f} (p={p_global:.2e})")
    print(f"  [{elapsed:.0f}s]")

    # Also evaluate the -core_n_heavy heuristic on same held-out set
    IDX_N_HEAVY = FEAT_NAMES.index("core_n_heavy")
    heuristic_scores = -X_test[:, IDX_N_HEAVY]
    m_heur = eval_metrics_for_target(heuristic_scores, y_test, g_test)
    rho_heur, _ = stats.spearmanr(heuristic_scores, y_test)
    print(f"\n  -core_n_heavy heuristic on same held-out set:")
    print(f"    NDCG@3: {m_heur['ndcg']:.4f}  Hit@1: {m_heur['hit_rate']:.3f}  "
          f"Spearman: {rho_heur:.4f}")

    return {
        "n_cores_total": len(all_cores),
        "n_cores_holdout": n_holdout,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "hgb": {
            "ndcg": m["ndcg"],
            "hit_rate": m["hit_rate"],
            "spearman": float(rho_global),
        },
        "heuristic": {
            "ndcg": m_heur["ndcg"],
            "hit_rate": m_heur["hit_rate"],
            "spearman": float(rho_heur),
        },
    }


def main():
    results = {}
    results["experiment_a"] = experiment_a()
    results["experiment_b"] = experiment_b()

    # Save
    out_path = Path("outputs/ood/novel_scaffold_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
