#!/usr/bin/env python
"""
Compute practical efficiency metrics for the Activity Cliffs paper.

Analysis 1: Position count distribution per molecule
Analysis 2: Expected rank to find the true most-sensitive position (model vs random)
Analysis 3: Hit@k curve (k=1,2,3) model vs random

Uses leave-one-target-out cross-validation to generate OOD predictions.
Writes results to outputs/practical_metrics.json.
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT / "evolve" / "eval_data" / "position_data.npz"
OUTPUT_PATH = PROJECT / "outputs" / "practical_metrics.json"

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


def main():
    t0 = time.perf_counter()

    # Load position data
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    groups = d["groups"]
    target_offsets = d["target_offsets"]
    target_names = d["target_names"]

    print(f"Data loaded: {X.shape[0]:,} positions, {len(target_names)} targets")

    # ── Leave-one-target-out predictions ──────────────────────────────────
    print("Running leave-one-target-out cross-validation...")
    ood_preds = np.zeros(len(y), dtype=np.float64)

    for i in range(len(target_names)):
        lo = target_offsets[i]
        hi = target_offsets[i + 1]

        train_mask = np.ones(len(y), dtype=bool)
        train_mask[lo:hi] = False

        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X[train_mask], y[train_mask])
        ood_preds[lo:hi] = model.predict(X[lo:hi])

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(target_names)} targets")

    elapsed_loo = time.perf_counter() - t0
    print(f"LOO-target predictions complete in {elapsed_loo:.0f}s")

    # ── Analysis 1: Position count distribution ──────────────────────────
    unique_mols, mol_counts = np.unique(groups, return_counts=True)
    n_mols = len(unique_mols)

    analysis1 = {
        "n_molecules": int(n_mols),
        "n_positions_total": int(len(y)),
        "mean": float(np.mean(mol_counts)),
        "median": float(np.median(mol_counts)),
        "p25": float(np.percentile(mol_counts, 25)),
        "p75": float(np.percentile(mol_counts, 75)),
        "min": int(np.min(mol_counts)),
        "max": int(np.max(mol_counts)),
        "std": float(np.std(mol_counts)),
    }

    freq = Counter(int(c) for c in mol_counts)
    freq_table = {}
    for k in sorted(freq.keys()):
        freq_table[str(k)] = int(freq[k])
    analysis1["frequency_table"] = freq_table

    print(f"\n===== Analysis 1: Position Count Distribution =====")
    print(f"Molecules: {n_mols:,}")
    print(f"Mean positions: {analysis1['mean']:.2f}")
    print(f"Median positions: {analysis1['median']:.1f}")
    print(f"IQR: [{analysis1['p25']:.1f}, {analysis1['p75']:.1f}]")
    print(f"Range: [{analysis1['min']}, {analysis1['max']}]")

    # ── Analysis 2 & 3: Rank and Hit@k ───────────────────────────────────
    model_ranks = []
    random_expected = []
    hit_at_k_model = {1: [], 2: [], 3: []}
    hit_at_k_random = {1: [], 2: [], 3: []}
    n_pos_list = []

    for mol_id in unique_mols:
        mask = groups == mol_id
        n_pos = int(mask.sum())
        if n_pos < 2:
            continue

        y_mol = y[mask]
        pred_mol = ood_preds[mask]

        # Handle ties in true labels: if there are multiple positions
        # tied for the best, count a hit if ANY of them appear at rank <= k
        true_best_val = np.max(y_mol)
        is_best = y_mol == true_best_val

        # Model ranking: sort positions by predicted SALI descending
        model_order = np.argsort(-pred_mol)

        # Rank of the first true-best position in model ordering
        best_rank = n_pos  # fallback
        for rank_idx, pos_idx in enumerate(model_order):
            if is_best[pos_idx]:
                best_rank = rank_idx + 1  # 1-indexed
                break

        model_ranks.append(best_rank)
        n_pos_list.append(n_pos)

        # Random expected rank = (n_pos + 1) / 2 when there's 1 best
        # With t tied best positions: E[rank] = (n_pos + 1) / (t + 1)
        n_tied = int(is_best.sum())
        random_exp = (n_pos + 1) / (n_tied + 1)
        random_expected.append(random_exp)

        # Hit@k
        for k in [1, 2, 3]:
            # Model: did model place a true-best in top k?
            top_k_indices = model_order[:k]
            hit = float(any(is_best[idx] for idx in top_k_indices))
            hit_at_k_model[k].append(hit)

            # Random: P(at least one of t best items in top k of n)
            # = 1 - C(n-t, k) / C(n, k) when k <= n-t, else 1.0
            if k >= n_pos:
                p_hit = 1.0
            elif n_tied >= n_pos:
                p_hit = 1.0
            else:
                # P(miss all) = C(n-t, k) / C(n, k) = prod_{i=0}^{k-1} (n-t-i)/(n-i)
                p_miss = 1.0
                for j in range(k):
                    if (n_pos - n_tied - j) <= 0:
                        p_miss = 0.0
                        break
                    p_miss *= (n_pos - n_tied - j) / (n_pos - j)
                p_hit = 1.0 - p_miss
            hit_at_k_random[k].append(p_hit)

    model_ranks = np.array(model_ranks)
    random_expected = np.array(random_expected)
    n_evaluated = len(model_ranks)

    analysis2 = {
        "n_molecules_evaluated": int(n_evaluated),
        "note": "molecules with >= 2 positions",
        "model_mean_rank": float(np.mean(model_ranks)),
        "model_median_rank": float(np.median(model_ranks)),
        "random_mean_expected_rank": float(np.mean(random_expected)),
        "random_median_expected_rank": float(np.median(random_expected)),
        "efficiency_gain_positions": float(np.mean(random_expected) - np.mean(model_ranks)),
        "efficiency_gain_fraction": float(1.0 - np.mean(model_ranks) / np.mean(random_expected)),
    }

    analysis3 = {
        "n_molecules_evaluated": int(n_evaluated),
    }
    for k in [1, 2, 3]:
        model_val = float(np.mean(hit_at_k_model[k]))
        random_val = float(np.mean(hit_at_k_random[k]))
        lift = model_val / random_val if random_val > 0 else float("inf")
        analysis3[f"model_hit_at_{k}"] = model_val
        analysis3[f"random_hit_at_{k}"] = random_val
        analysis3[f"lift_at_{k}"] = lift

    print(f"\n===== Analysis 2: Expected Rank to Find Top Position =====")
    print(f"Molecules evaluated (>= 2 positions): {n_evaluated:,}")
    print(f"Model mean rank: {analysis2['model_mean_rank']:.3f}")
    print(f"Random mean expected rank: {analysis2['random_mean_expected_rank']:.3f}")
    print(f"Efficiency gain: {analysis2['efficiency_gain_positions']:.3f} fewer positions to test")
    print(f"Efficiency gain: {analysis2['efficiency_gain_fraction']:.1%} reduction")

    print(f"\n===== Analysis 3: Hit@k Curve =====")
    print(f"{'':>10s}  {'Model':>8s}  {'Random':>8s}  {'Lift':>6s}")
    for k in [1, 2, 3]:
        m = analysis3[f"model_hit_at_{k}"]
        r = analysis3[f"random_hit_at_{k}"]
        l = analysis3[f"lift_at_{k}"]
        print(f"  Hit@{k}:    {m:8.4f}  {r:8.4f}  {l:5.2f}x")

    # ── Also compute for molecules with >= 3 positions (NDCG@3 cohort) ──
    mask_ge3 = np.array(n_pos_list) >= 3
    if mask_ge3.sum() > 0:
        analysis2["ge3_model_mean_rank"] = float(np.mean(model_ranks[mask_ge3]))
        analysis2["ge3_random_mean_expected_rank"] = float(np.mean(random_expected[mask_ge3]))
        analysis2["ge3_efficiency_gain_positions"] = float(
            np.mean(random_expected[mask_ge3]) - np.mean(model_ranks[mask_ge3])
        )
        analysis2["ge3_efficiency_gain_fraction"] = float(
            1.0 - np.mean(model_ranks[mask_ge3]) / np.mean(random_expected[mask_ge3])
        )
        analysis2["ge3_n_molecules"] = int(mask_ge3.sum())

        print(f"\n  (Restricted to molecules with >= 3 positions: {int(mask_ge3.sum()):,})")
        print(f"  Model mean rank: {analysis2['ge3_model_mean_rank']:.3f}")
        print(f"  Random mean expected rank: {analysis2['ge3_random_mean_expected_rank']:.3f}")
        print(f"  Efficiency gain: {analysis2['ge3_efficiency_gain_positions']:.3f} positions")
        print(f"  Efficiency gain: {analysis2['ge3_efficiency_gain_fraction']:.1%}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "analysis_description": "Practical efficiency metrics for Activity Cliffs SALI model",
        "evaluation_method": "Leave-one-target-out (50 targets, OOD predictions)",
        "sali_definition": "|delta_pActivity| / (max_rgroup_n_heavy + 1)",
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "position_count_distribution": analysis1,
        "expected_rank_to_find_top": analysis2,
        "hit_at_k": analysis3,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
