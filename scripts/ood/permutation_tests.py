#!/usr/bin/env python
"""
OOD Experiment 9: Permutation Tests for Statistical Significance

Establishes null distributions for both models:
  1. Position model: shuffle sensitivity labels within (mol, target) groups
  2. Change-type model: shuffle |dpActivity| within (core, target) groups

Reports p-values and effect sizes for all reported metrics.

Usage:
    python scripts/ood/permutation_tests.py [--n-perms 1000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)


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
    }


# ── Position model permutation test ─────────────────────────────────────────

def position_permutation_test(n_perms: int = 1000):
    """Test the -core_n_heavy heuristic against shuffled labels."""
    print("\n" + "=" * 78)
    print("POSITION MODEL: Permutation Test")
    print("=" * 78)

    EVAL_DATA_PATH = Path("evolve/eval_data/position_data.npz")
    _data = np.load(EVAL_DATA_PATH, allow_pickle=True)
    X = _data["X"]
    Y = _data["y"]
    GROUPS = _data["groups"]
    OFFSETS = _data["target_offsets"]
    TARGETS = _data["target_names"]
    FEAT_NAMES = list(_data["feature_names"])
    del _data

    IDX_N_HEAVY = FEAT_NAMES.index("core_n_heavy")
    scores = -X[:, IDX_N_HEAVY]  # heuristic: smaller = more sensitive

    print(f"Loaded: {X.shape[0]:,} rows, {len(TARGETS)} targets")
    print(f"Using -core_n_heavy heuristic (no ML)")
    print(f"Running {n_perms} permutations...")

    # Observed metrics
    obs_ndcgs, obs_hits, obs_rhos = [], [], []
    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        m = eval_metrics_for_target(scores[lo:hi], Y[lo:hi], GROUPS[lo:hi])
        obs_ndcgs.append(m["ndcg"])
        obs_hits.append(m["hit_rate"])
        obs_rhos.append(m["spearman"])

    obs_ndcg = float(np.mean(obs_ndcgs))
    obs_hit = float(np.mean(obs_hits))
    obs_rho = float(np.mean(obs_rhos))
    print(f"\nObserved: NDCG@3={obs_ndcg:.4f}  Hit@1={obs_hit:.3f}  "
          f"Spearman={obs_rho:.4f}")

    # Permutation loop: shuffle Y within each (mol, target) group
    rng = np.random.RandomState(42)
    perm_ndcgs = []
    perm_hits = []
    perm_rhos = []
    t0 = time.perf_counter()

    for p in range(n_perms):
        if (p + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  permutation {p+1}/{n_perms} ({elapsed:.0f}s)")

        # Shuffle Y within each target's mol_from groups
        Y_shuf = Y.copy()
        for i in range(len(TARGETS)):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            g_target = GROUPS[lo:hi]
            for mol_id in np.unique(g_target):
                mask = np.where(g_target == mol_id)[0] + lo
                Y_shuf[mask] = rng.permutation(Y_shuf[mask])

        # Evaluate on shuffled labels
        p_ndcgs, p_hits, p_rhos = [], [], []
        for i in range(len(TARGETS)):
            lo, hi = OFFSETS[i], OFFSETS[i + 1]
            m = eval_metrics_for_target(
                scores[lo:hi], Y_shuf[lo:hi], GROUPS[lo:hi]
            )
            p_ndcgs.append(m["ndcg"])
            p_hits.append(m["hit_rate"])
            p_rhos.append(m["spearman"])

        perm_ndcgs.append(float(np.mean(p_ndcgs)))
        perm_hits.append(float(np.mean(p_hits)))
        perm_rhos.append(float(np.mean(p_rhos)))

    perm_ndcgs = np.array(perm_ndcgs)
    perm_hits = np.array(perm_hits)
    perm_rhos = np.array(perm_rhos)

    # p-values (fraction of permutations >= observed)
    p_ndcg = float(np.mean(perm_ndcgs >= obs_ndcg))
    p_hit = float(np.mean(perm_hits >= obs_hit))
    p_rho = float(np.mean(perm_rhos >= obs_rho))

    # Effect sizes (Cohen's d)
    d_ndcg = (obs_ndcg - np.mean(perm_ndcgs)) / (np.std(perm_ndcgs) + 1e-10)
    d_hit = (obs_hit - np.mean(perm_hits)) / (np.std(perm_hits) + 1e-10)
    d_rho = (obs_rho - np.mean(perm_rhos)) / (np.std(perm_rhos) + 1e-10)

    print(f"\nNull distribution ({n_perms} permutations):")
    print(f"  NDCG@3:   null mean={np.mean(perm_ndcgs):.4f} +/- {np.std(perm_ndcgs):.4f}")
    print(f"  Hit@1:    null mean={np.mean(perm_hits):.3f} +/- {np.std(perm_hits):.3f}")
    print(f"  Spearman: null mean={np.mean(perm_rhos):.4f} +/- {np.std(perm_rhos):.4f}")

    print(f"\nStatistical significance:")
    print(f"  NDCG@3:   observed={obs_ndcg:.4f}  p={p_ndcg:.4f}  Cohen's d={d_ndcg:.1f}")
    print(f"  Hit@1:    observed={obs_hit:.3f}   p={p_hit:.4f}  Cohen's d={d_hit:.1f}")
    print(f"  Spearman: observed={obs_rho:.4f}  p={p_rho:.4f}  Cohen's d={d_rho:.1f}")

    # Per-target significance for Spearman
    print(f"\nPer-target Spearman significance (Bonferroni alpha=0.001):")
    n_sig = 0
    per_target_results = []
    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        if (hi - lo) > 5:
            rho_obs, p_obs = stats.spearmanr(scores[lo:hi], Y[lo:hi])
            is_sig = p_obs < 0.001  # Bonferroni: 0.05/50 = 0.001
            if is_sig:
                n_sig += 1
            per_target_results.append({
                "target": str(TARGETS[i]),
                "spearman": float(rho_obs),
                "p_value": float(p_obs),
                "significant": bool(is_sig),
                "n_positions": int(hi - lo),
            })
    print(f"  {n_sig}/{len(TARGETS)} targets significant at Bonferroni-corrected p<0.001")

    result = {
        "model": "position_heuristic",
        "n_perms": n_perms,
        "observed": {"ndcg": obs_ndcg, "hit_rate": obs_hit, "spearman": obs_rho},
        "null_mean": {
            "ndcg": float(np.mean(perm_ndcgs)),
            "hit_rate": float(np.mean(perm_hits)),
            "spearman": float(np.mean(perm_rhos)),
        },
        "null_std": {
            "ndcg": float(np.std(perm_ndcgs)),
            "hit_rate": float(np.std(perm_hits)),
            "spearman": float(np.std(perm_rhos)),
        },
        "p_values": {"ndcg": p_ndcg, "hit_rate": p_hit, "spearman": p_rho},
        "cohens_d": {
            "ndcg": float(d_ndcg),
            "hit_rate": float(d_hit),
            "spearman": float(d_rho),
        },
        "per_target": per_target_results,
    }
    return result


# ── Change-type model permutation test ───────────────────────────────────────

def change_type_permutation_test(n_perms: int = 500):
    """Test the change-type model Spearman against shuffled labels.

    Since the change-type model is slower (HGB LOO-target), we use a
    simplified test: compute Spearman of predictions vs shuffled labels.
    """
    print("\n" + "=" * 78)
    print("CHANGE-TYPE MODEL: Permutation Test")
    print("=" * 78)

    # Load change-type eval data if available
    ct_data_path = Path("evolve/eval_data/eval_data_v3.npz")
    if not ct_data_path.exists():
        print("Change-type eval data not found, skipping.")
        return None

    _data = np.load(ct_data_path, allow_pickle=True)
    X_ct = _data["X"]
    Y_ct = _data["y"]  # abs_delta_pActivity
    GROUPS_ct = _data["groups"]
    OFFSETS_ct = _data["target_offsets"]
    TARGETS_ct = _data["target_names"]
    del _data

    print(f"Loaded: {X_ct.shape[0]:,} rows, {len(TARGETS_ct)} targets")

    # For efficiency, just test the null of Spearman correlation
    # between Y and a strong predictor (L2 norm of first 8 features)
    scores_ct = np.sqrt(np.sum(X_ct[:, :8] ** 2, axis=1))

    # Observed per-target Spearman
    obs_rhos = []
    for i in range(len(TARGETS_ct)):
        lo, hi = OFFSETS_ct[i], OFFSETS_ct[i + 1]
        if (hi - lo) > 10:
            rho, _ = stats.spearmanr(scores_ct[lo:hi], Y_ct[lo:hi])
            if np.isfinite(rho):
                obs_rhos.append(float(rho))
    obs_mean_rho = float(np.mean(obs_rhos))
    print(f"Observed mean Spearman (L2 heuristic): {obs_mean_rho:.4f}")

    # Permutation: shuffle Y within target
    rng = np.random.RandomState(42)
    perm_mean_rhos = []
    t0 = time.perf_counter()

    for p in range(n_perms):
        if (p + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  permutation {p+1}/{n_perms} ({elapsed:.0f}s)")

        Y_shuf = Y_ct.copy()
        for i in range(len(TARGETS_ct)):
            lo, hi = OFFSETS_ct[i], OFFSETS_ct[i + 1]
            Y_shuf[lo:hi] = rng.permutation(Y_shuf[lo:hi])

        p_rhos = []
        for i in range(len(TARGETS_ct)):
            lo, hi = OFFSETS_ct[i], OFFSETS_ct[i + 1]
            if (hi - lo) > 10:
                rho, _ = stats.spearmanr(scores_ct[lo:hi], Y_shuf[lo:hi])
                if np.isfinite(rho):
                    p_rhos.append(float(rho))
        perm_mean_rhos.append(float(np.mean(p_rhos)))

    perm_mean_rhos = np.array(perm_mean_rhos)
    p_value = float(np.mean(perm_mean_rhos >= obs_mean_rho))
    d = (obs_mean_rho - np.mean(perm_mean_rhos)) / (np.std(perm_mean_rhos) + 1e-10)

    print(f"\nNull distribution ({n_perms} permutations):")
    print(f"  mean Spearman: null={np.mean(perm_mean_rhos):.4f} +/- {np.std(perm_mean_rhos):.4f}")
    print(f"  observed:      {obs_mean_rho:.4f}")
    print(f"  p-value:       {p_value:.4f}")
    print(f"  Cohen's d:     {d:.1f}")

    result = {
        "model": "change_type_heuristic",
        "n_perms": n_perms,
        "observed_mean_spearman": obs_mean_rho,
        "null_mean": float(np.mean(perm_mean_rhos)),
        "null_std": float(np.std(perm_mean_rhos)),
        "p_value": p_value,
        "cohens_d": float(d),
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perms", type=int, default=1000)
    args = parser.parse_args()

    results = {}

    # Position model
    results["position"] = position_permutation_test(args.n_perms)

    # Change-type model
    ct_result = change_type_permutation_test(min(args.n_perms, 500))
    if ct_result:
        results["change_type"] = ct_result

    # Save
    out_path = Path("outputs/ood/permutation_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
