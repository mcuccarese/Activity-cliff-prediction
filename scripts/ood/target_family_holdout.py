#!/usr/bin/env python
"""
OOD Experiment 3: Target Family Holdout

Tests whether the model generalizes across protein families, not just
individual targets.

Protocol:
  1. Classify 50 targets into protein families
  2. Leave-one-family-out: train on all targets except family F,
     test on family F
  3. Special test: leave ALL kinases out (22 targets, 44% of data)

Usage:
    python scripts/ood/target_family_holdout.py
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

# ── Target family classification ─────────────────────────────────────────────
# Based on ChEMBL target info and UniProt classification

TARGET_FAMILIES = {
    # Kinases (22 targets)
    "kinase": [
        "CHEMBL203",    # EGFR
        "CHEMBL2041",   # Ret
        "CHEMBL5251",   # BTK
        "CHEMBL2971",   # JAK2
        "CHEMBL279",    # VEGFR2 (KDR)
        "CHEMBL2835",   # JAK1
        "CHEMBL2148",   # JAK3
        "CHEMBL5145",   # B-Raf
        "CHEMBL3778",   # IRAK4
        "CHEMBL4040",   # MEK1/MAP2K1
        "CHEMBL2599",   # SYK
        "CHEMBL3717",   # c-Met/HGF receptor
        "CHEMBL3553",   # TYK2
        "CHEMBL1974",   # FLT3
        "CHEMBL2842",   # mTOR
        "CHEMBL2147",   # Pim-1
        "CHEMBL267",    # Src
        "CHEMBL4282",   # Rac-alpha/Akt
        "CHEMBL2973",   # ROCK2
        "CHEMBL3130",   # PI3Kd
        "CHEMBL4005",   # PI3Ka
        "CHEMBL2815",   # TrkA/NGF receptor
    ],
    # Epigenetic (4 targets)
    "epigenetic": [
        "CHEMBL1163125",  # BRD4
        "CHEMBL325",      # HDAC1
        "CHEMBL1865",     # HDAC6
        "CHEMBL6136",     # KDM1A/LSD1
    ],
    # Enzymes (non-kinase) (7 targets)
    "enzyme": [
        "CHEMBL220",      # AChE
        "CHEMBL2039",     # MAO-B
        "CHEMBL4409",     # PDE10A
        "CHEMBL340",      # CYP3A4
        "CHEMBL4822",     # BACE1/beta-secretase
        "CHEMBL2007625",  # IDH2
        "CHEMBL1744525",  # NAMPT
    ],
    # Ion channels (3 targets)
    "ion_channel": [
        "CHEMBL240",   # hERG/KCNH2
        "CHEMBL4296",  # Nav1.9
        "CHEMBL2998",  # P2X3
    ],
    # Immune/inflammatory (7 targets)
    "immune": [
        "CHEMBL1741186",  # RORgamma
        "CHEMBL5936",     # TLR7
        "CHEMBL5805",     # TLR8
        "CHEMBL5804",     # TLR9
        "CHEMBL4685",     # IDO1
        "CHEMBL4805",     # P2X7
        "CHEMBL3650",     # SHP2/PTPN11
    ],
    # Receptors/other (7 targets)
    "receptor_other": [
        "CHEMBL206",   # ER-alpha
        "CHEMBL260",   # MAP p38a
        "CHEMBL230",   # COX-2
        "CHEMBL5023",  # MDM2
        "CHEMBL5113",  # OX1R
        "CHEMBL4792",  # OX2R
        "CHEMBL284",   # Factor Xa
    ],
}

# Verify all targets are assigned
ALL_ASSIGNED = set()
for fam, targets in TARGET_FAMILIES.items():
    ALL_ASSIGNED.update(targets)


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

# Build target -> index mapping
TARGET_TO_IDX = {str(t): i for i, t in enumerate(TARGETS)}

# Check which targets aren't classified
unclassified = []
for t in TARGETS:
    if str(t) not in ALL_ASSIGNED:
        unclassified.append(str(t))

if unclassified:
    print(f"WARNING: {len(unclassified)} targets not in any family: {unclassified}")
    # Add them to "other"
    TARGET_FAMILIES.setdefault("unclassified", []).extend(unclassified)

print(f"Loaded: {X.shape[0]:,} rows, {len(TARGETS)} targets")
print(f"\nTarget family sizes:")
for fam, targets in sorted(TARGET_FAMILIES.items()):
    present = [t for t in targets if t in TARGET_TO_IDX]
    n_positions = sum(
        OFFSETS[TARGET_TO_IDX[t] + 1] - OFFSETS[TARGET_TO_IDX[t]]
        for t in present
    )
    print(f"  {fam:20s}: {len(present):>2d} targets, {n_positions:>8,} positions")

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
    rho = 0.0
    if len(y) > 5:
        r, _ = stats.spearmanr(scores, y)
        if np.isfinite(r):
            rho = float(r)
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": rho,
        "n_groups": len(ndcgs),
    }


# ── Leave-one-family-out ─────────────────────────────────────────────────────

def leave_one_family_out():
    print("\n" + "=" * 78)
    print("LEAVE-ONE-FAMILY-OUT EVALUATION")
    print("=" * 78)

    results = {}

    for fam_name, fam_targets in sorted(TARGET_FAMILIES.items()):
        fam_indices = [TARGET_TO_IDX[t] for t in fam_targets if t in TARGET_TO_IDX]
        if not fam_indices:
            continue

        # Build test mask (all targets in this family)
        test_mask = np.zeros(len(Y), dtype=bool)
        for idx in fam_indices:
            lo, hi = OFFSETS[idx], OFFSETS[idx + 1]
            test_mask[lo:hi] = True

        train_mask = ~test_mask

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test < 50:
            print(f"  {fam_name}: too few test positions ({n_test}), skipping")
            continue

        X_train, y_train = X[train_mask], Y[train_mask]
        X_test, y_test = X[test_mask], Y[test_mask]
        g_test = GROUPS[test_mask]

        t0 = time.perf_counter()
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.perf_counter() - t0

        # Evaluate per target within the family
        per_target_ndcgs = []
        for idx in fam_indices:
            lo, hi = OFFSETS[idx], OFFSETS[idx + 1]
            m = eval_metrics_for_target(preds[lo - test_mask[:lo].sum():hi - test_mask[:lo].sum()],
                                         y_test[lo - test_mask[:lo].sum():hi - test_mask[:lo].sum()],
                                         g_test[lo - test_mask[:lo].sum():hi - test_mask[:lo].sum()])
            # Actually, the masking is tricky with test_mask indexing. Let's do it differently.
            pass

        # Evaluate on full family test set
        m = eval_metrics_for_target(preds, y_test, g_test)

        # Also evaluate heuristic
        IDX_N_HEAVY = FEAT_NAMES.index("core_n_heavy")
        heur_scores = -X_test[:, IDX_N_HEAVY]
        m_heur = eval_metrics_for_target(heur_scores, y_test, g_test)

        print(f"  {fam_name:20s} ({len(fam_indices):>2d} targets, {n_test:>7,} test)  "
              f"HGB NDCG@3={m['ndcg']:.4f}  Hit@1={m['hit_rate']:.3f}  "
              f"Spearman={m['spearman']:.4f}  |  "
              f"Heuristic NDCG@3={m_heur['ndcg']:.4f}  [{elapsed:.0f}s]")

        results[fam_name] = {
            "n_targets": len(fam_indices),
            "targets": [str(TARGETS[i]) for i in fam_indices],
            "n_train": int(n_train),
            "n_test": int(n_test),
            "hgb": {
                "ndcg": m["ndcg"],
                "hit_rate": m["hit_rate"],
                "spearman": m["spearman"],
            },
            "heuristic": {
                "ndcg": m_heur["ndcg"],
                "hit_rate": m_heur["hit_rate"],
                "spearman": m_heur["spearman"],
            },
        }

    return results


def main():
    results = leave_one_family_out()

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY: LEAVE-ONE-FAMILY-OUT")
    print("=" * 78)
    print(f"{'Family':<20s} {'N targets':>10s} {'HGB NDCG@3':>12s} {'Heur NDCG@3':>13s} {'HGB-Heur':>10s}")
    print("-" * 70)
    for fam, r in sorted(results.items()):
        gap = r["hgb"]["ndcg"] - r["heuristic"]["ndcg"]
        print(f"  {fam:<18s} {r['n_targets']:>10d} {r['hgb']['ndcg']:>12.4f} "
              f"{r['heuristic']['ndcg']:>13.4f} {gap:>+10.4f}")

    ndcgs = [r["hgb"]["ndcg"] for r in results.values()]
    print(f"\n  Mean NDCG@3 across families: {np.mean(ndcgs):.4f} +/- {np.std(ndcgs):.4f}")

    baseline = 0.9593  # LOO-target result
    print(f"  LOO-target baseline:         {baseline:.4f}")
    print(f"  Mean family holdout gap:     {np.mean(ndcgs) - baseline:+.4f}")

    # Save
    out_path = Path("outputs/ood/target_family_holdout_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
