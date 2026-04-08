#!/usr/bin/env python
"""
SALI-Normalized OOD Evaluations

Runs the three key OOD stress tests (protein family holdout, novel scaffold
holdout, temporal split) using SALI-normalized sensitivity as the target
variable instead of raw sensitivity.

This mirrors the raw evaluations in:
  - target_family_holdout.py
  - novel_scaffold_holdout.py
  - temporal_split.py

but replaces y = mean(|delta_pActivity|) with
         y = mean(|delta_pActivity| / (max_rgroup_n_heavy + 1))

Usage:
    python scripts/ood/sali_ood_evaluations.py [--chembl-sqlite PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent.parent
MMPS_PATH = PROJECT / "outputs" / "mmps" / "all_mmps.parquet"
CTX_PATH = PROJECT / "outputs" / "features" / "context_3d.parquet"
OUTPUT_DIR = PROJECT / "outputs" / "ood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB = r"D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}

# ── Target families (same as target_family_holdout.py) ─────────────────────
TARGET_FAMILIES = {
    "kinase": [
        "CHEMBL203", "CHEMBL2041", "CHEMBL5251", "CHEMBL2971", "CHEMBL279",
        "CHEMBL2835", "CHEMBL2148", "CHEMBL5145", "CHEMBL3778", "CHEMBL4040",
        "CHEMBL2599", "CHEMBL3717", "CHEMBL3553", "CHEMBL1974", "CHEMBL2842",
        "CHEMBL2147", "CHEMBL267", "CHEMBL4282", "CHEMBL2973", "CHEMBL3130",
        "CHEMBL4005", "CHEMBL2815",
    ],
    "epigenetic": ["CHEMBL1163125", "CHEMBL325", "CHEMBL1865", "CHEMBL6136"],
    "enzyme": [
        "CHEMBL220", "CHEMBL2039", "CHEMBL4409", "CHEMBL340",
        "CHEMBL4822", "CHEMBL2007625", "CHEMBL1744525",
    ],
    "ion_channel": ["CHEMBL240", "CHEMBL4296", "CHEMBL2998"],
    "immune": [
        "CHEMBL1741186", "CHEMBL5936", "CHEMBL5805", "CHEMBL5804",
        "CHEMBL4685", "CHEMBL4805", "CHEMBL3650",
    ],
    "receptor_other": [
        "CHEMBL206", "CHEMBL260", "CHEMBL230", "CHEMBL5023",
        "CHEMBL5113", "CHEMBL4792", "CHEMBL284",
    ],
}


# ── Utility functions ──────────────────────────────────────────────────────

def _n_heavy(smi: str) -> int:
    if not smi or smi == "[H]":
        return 0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


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


def eval_metrics_grouped(scores, y, groups, k=3):
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


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: Build SALI position-level data
# ══════════════════════════════════════════════════════════════════════════

def build_sali_position_data():
    """Load MMPs, compute SALI, aggregate to positions, build feature matrix.

    Returns (X, feature_names, groups, target_offsets, target_names,
             y_sali, core_smiles_arr)
    """
    logger.info("PHASE 1: Building SALI position-level data")

    # Load MMPs
    logger.info("  Loading MMPs...")
    mmps = pd.read_parquet(
        MMPS_PATH,
        columns=[
            "target_chembl_id", "mol_from", "core_smiles",
            "rgroup_from", "rgroup_to", "abs_delta_pActivity",
        ],
    )
    logger.info("  %s MMPs loaded", f"{len(mmps):,}")

    # Compute R-group heavy atom counts
    logger.info("  Computing R-group heavy atom counts...")
    t0 = time.perf_counter()
    all_rgroups = set(mmps["rgroup_from"].unique()) | set(mmps["rgroup_to"].unique())
    rg_cache: dict[str, int] = {}
    for i, rg in enumerate(all_rgroups):
        rg_cache[rg] = _n_heavy(rg)
        if (i + 1) % 50000 == 0:
            logger.info("    ... %d/%d R-groups", i + 1, len(all_rgroups))
    logger.info("  R-group sizes computed in %.0fs", time.perf_counter() - t0)

    mmps["max_rg_nheavy"] = np.maximum(
        mmps["rgroup_from"].map(rg_cache).astype(np.float32),
        mmps["rgroup_to"].map(rg_cache).astype(np.float32),
    )
    mmps["sali"] = mmps["abs_delta_pActivity"] / (mmps["max_rg_nheavy"] + 1.0)

    # Aggregate to position level
    logger.info("  Aggregating to position level...")
    MIN_MMPS, MIN_POS = 3, 3

    pos = (
        mmps.groupby(["core_smiles", "target_chembl_id"])
        .agg(sali_mean=("sali", "mean"), n_mmps=("abs_delta_pActivity", "count"))
        .reset_index()
    )
    pos = pos[pos["n_mmps"] >= MIN_MMPS].reset_index(drop=True)

    mol_pos = (
        mmps[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
        .merge(pos, on=["core_smiles", "target_chembl_id"])
    )
    pos_count = (
        mol_pos.groupby(["mol_from", "target_chembl_id"]).size()
        .reset_index(name="n_pos")
    )
    pos_count = pos_count[pos_count["n_pos"] >= MIN_POS]
    mol_pos = mol_pos.merge(
        pos_count[["mol_from", "target_chembl_id"]],
        on=["mol_from", "target_chembl_id"],
    )

    # Load 3D context features
    logger.info("  Loading 3D context features...")
    ctx_df = pd.read_parquet(CTX_PATH)
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(
        zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32))
    )

    # Core topology
    unique_cores = mol_pos["core_smiles"].unique()
    core_topo = {}
    for smi in unique_cores:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            nh = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)
            nr = mol.GetRingInfo().NumRings()
        else:
            nh, nr = 0, 0
        core_topo[smi] = (nh, nr)

    # Sort and build arrays
    mol_pos = mol_pos.sort_values(
        ["target_chembl_id", "mol_from", "core_smiles"]
    ).reset_index(drop=True)

    target_names = []
    target_offsets = [0]
    for target, tdf in mol_pos.groupby("target_chembl_id", sort=True):
        target_names.append(str(target))
        target_offsets.append(target_offsets[-1] + len(tdf))
    target_offsets = np.array(target_offsets)
    target_names = np.array(target_names)

    n_ctx = len(ctx_cols)
    zero_ctx = np.zeros(n_ctx, dtype=np.float32)
    X_list = []
    for smi in mol_pos["core_smiles"]:
        ctx = ctx_lookup.get(smi, zero_ctx)
        nh, nr = core_topo.get(smi, (0, 0))
        X_list.append(np.concatenate([ctx, [float(nh), float(nr)]]))

    X = np.array(X_list, dtype=np.float32)
    feature_names = [f"ctx_{c}" for c in ctx_cols] + ["core_n_heavy", "core_n_rings"]
    groups = mol_pos["mol_from"].values.astype(np.int64)
    y_sali = mol_pos["sali_mean"].values.astype(np.float32)
    core_smiles_arr = mol_pos["core_smiles"].values

    logger.info("  Built: X=%s, %d targets, %d positions",
                X.shape, len(target_names), len(y_sali))

    return X, feature_names, groups, target_offsets, target_names, y_sali, core_smiles_arr


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION 1: Target Family Holdout (SALI)
# ══════════════════════════════════════════════════════════════════════════

def eval_family_holdout(X, feat_names, groups, offsets, targets, y):
    logger.info("\n" + "=" * 78)
    logger.info("FAMILY HOLDOUT (SALI TARGET)")
    logger.info("=" * 78)

    target_to_idx = {str(t): i for i, t in enumerate(targets)}
    IDX_N_HEAVY = feat_names.index("core_n_heavy")
    results = {}

    for fam_name, fam_targets in sorted(TARGET_FAMILIES.items()):
        fam_indices = [target_to_idx[t] for t in fam_targets if t in target_to_idx]
        if not fam_indices:
            continue

        test_mask = np.zeros(len(y), dtype=bool)
        for idx in fam_indices:
            lo, hi = offsets[idx], offsets[idx + 1]
            test_mask[lo:hi] = True

        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        g_test = groups[test_mask]

        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m = eval_metrics_grouped(preds, y_test, g_test)
        m_heur = eval_metrics_grouped(-X_test[:, IDX_N_HEAVY], y_test, g_test)

        logger.info("  %-20s (%2d targets)  HGB=%.4f  Heur=%.4f",
                     fam_name, len(fam_indices), m["ndcg"], m_heur["ndcg"])

        results[fam_name] = {
            "n_targets": len(fam_indices),
            "targets": [str(targets[i]) for i in fam_indices],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "hgb": {"ndcg": m["ndcg"], "hit_rate": m["hit_rate"], "spearman": m["spearman"]},
            "heuristic": {"ndcg": m_heur["ndcg"], "hit_rate": m_heur["hit_rate"], "spearman": m_heur["spearman"]},
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION 2: Novel Scaffold Holdout (SALI)
# ══════════════════════════════════════════════════════════════════════════

def eval_novel_scaffold(X, feat_names, groups, offsets, targets, y, core_smiles):
    logger.info("\n" + "=" * 78)
    logger.info("NOVEL SCAFFOLD HOLDOUT (SALI TARGET)")
    logger.info("=" * 78)

    IDX_N_HEAVY = feat_names.index("core_n_heavy")

    # Build core -> targets mapping
    core_to_targets: dict[str, set[str]] = defaultdict(set)
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        for smi in core_smiles[lo:hi]:
            core_to_targets[smi].add(str(targets[i]))

    # --- Experiment A: LOO-target with seen/unseen core breakdown ---
    logger.info("  Experiment A: LOO-target with seen/unseen breakdown")
    results_seen, results_unseen = [], []

    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        X_test, y_test = X[lo:hi], y[lo:hi]
        g_test = groups[lo:hi]
        cores_test = core_smiles[lo:hi]

        train_mask = np.ones(len(y), dtype=bool)
        train_mask[lo:hi] = False
        train_cores = set(core_smiles[train_mask])

        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X_test)

        seen_mask = np.array([c in train_cores for c in cores_test])
        unseen_mask = ~seen_mask

        if seen_mask.sum() > 10:
            m = eval_metrics_grouped(preds[seen_mask], y_test[seen_mask], g_test[seen_mask])
            results_seen.append(m["ndcg"])
        if unseen_mask.sum() > 10:
            m = eval_metrics_grouped(preds[unseen_mask], y_test[unseen_mask], g_test[unseen_mask])
            results_unseen.append(m["ndcg"])

        if (i + 1) % 10 == 0:
            logger.info("    ... %d/%d targets", i + 1, len(targets))

    exp_a = {
        "seen_ndcg_mean": float(np.mean(results_seen)) if results_seen else None,
        "unseen_ndcg_mean": float(np.mean(results_unseen)) if results_unseen else None,
        "n_targets_with_seen": len(results_seen),
        "n_targets_with_unseen": len(results_unseen),
    }
    logger.info("  Exp A: Seen=%.4f  Unseen=%.4f",
                exp_a["seen_ndcg_mean"] or 0, exp_a["unseen_ndcg_mean"] or 0)

    # --- Experiment B: Scaffold-level holdout (20% cores held out) ---
    logger.info("  Experiment B: 20%% scaffold holdout")
    rng = np.random.RandomState(42)
    all_cores = list(set(core_smiles))
    n_holdout = int(len(all_cores) * 0.2)
    holdout_cores = set(rng.choice(all_cores, size=n_holdout, replace=False))

    holdout_mask = np.array([c in holdout_cores for c in core_smiles])
    train_mask = ~holdout_mask

    model = HistGradientBoostingRegressor(**HGB_KWARGS)
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[holdout_mask])

    m_hgb = eval_metrics_grouped(preds, y[holdout_mask], groups[holdout_mask])
    m_heur = eval_metrics_grouped(-X[holdout_mask, IDX_N_HEAVY], y[holdout_mask], groups[holdout_mask])

    exp_b = {
        "hgb": {"ndcg": m_hgb["ndcg"], "hit_rate": m_hgb["hit_rate"], "spearman": m_hgb["spearman"]},
        "heuristic": {"ndcg": m_heur["ndcg"], "hit_rate": m_heur["hit_rate"], "spearman": m_heur["spearman"]},
    }
    logger.info("  Exp B: HGB=%.4f  Heur=%.4f", m_hgb["ndcg"], m_heur["ndcg"])

    return {"experiment_a": exp_a, "experiment_b": exp_b}


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION 3: Temporal Split (SALI)
# ══════════════════════════════════════════════════════════════════════════

def eval_temporal_split(db_path: str):
    logger.info("\n" + "=" * 78)
    logger.info("TEMPORAL SPLIT (SALI TARGET)")
    logger.info("=" * 78)

    if not Path(db_path).exists():
        logger.warning("ChEMBL DB not found at %s — skipping temporal split", db_path)
        return None

    # Get document years
    logger.info("  Querying ChEMBL for document years...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT a.molregno,
           MIN(d.year) AS first_year,
           MAX(d.year) AS last_year
    FROM activities a
    JOIN assays ass ON a.assay_id = ass.assay_id
    JOIN docs d ON ass.doc_id = d.doc_id
    WHERE d.year IS NOT NULL
      AND a.standard_type = 'IC50'
      AND a.standard_units = 'nM'
      AND ass.confidence_score >= 7
    GROUP BY a.molregno
    """
    mol_years = pd.read_sql_query(query, conn)
    conn.close()
    mol_year_map = dict(zip(mol_years["molregno"], mol_years["first_year"]))

    # Load MMPs with R-group info for SALI
    logger.info("  Loading MMPs...")
    mmps = pd.read_parquet(
        MMPS_PATH,
        columns=[
            "target_chembl_id", "mol_from", "mol_to", "core_smiles",
            "rgroup_from", "rgroup_to", "abs_delta_pActivity",
        ],
    )

    # Compute SALI at MMP level
    logger.info("  Computing SALI normalization...")
    all_rgroups = set(mmps["rgroup_from"].unique()) | set(mmps["rgroup_to"].unique())
    rg_cache = {rg: _n_heavy(rg) for rg in all_rgroups}
    mmps["max_rg_nheavy"] = np.maximum(
        mmps["rgroup_from"].map(rg_cache).astype(np.float32),
        mmps["rgroup_to"].map(rg_cache).astype(np.float32),
    )
    mmps["sali"] = mmps["abs_delta_pActivity"] / (mmps["max_rg_nheavy"] + 1.0)

    # Add years
    mmps["year_from"] = mmps["mol_from"].map(mol_year_map)
    mmps["year_to"] = mmps["mol_to"].map(mol_year_map)
    mmps["mmp_year"] = mmps[["year_from", "year_to"]].max(axis=1)
    mmps = mmps.dropna(subset=["mmp_year"]).reset_index(drop=True)
    mmps["mmp_year"] = mmps["mmp_year"].astype(int)

    # Load 3D context features
    ctx_df = pd.read_parquet(CTX_PATH)
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(
        zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32))
    )
    zero_ctx = np.zeros(len(ctx_cols), dtype=np.float32)

    def _core_n_heavy(smi):
        mol = Chem.MolFromSmiles(smi)
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0) if mol else 0

    def _core_n_rings(smi):
        mol = Chem.MolFromSmiles(smi)
        return mol.GetRingInfo().NumRings() if mol else 0

    results = {}
    for cutoff in [2015, 2018]:
        logger.info("  Cutoff: %d", cutoff)
        mmps_train = mmps[mmps["mmp_year"] <= cutoff]
        mmps_test = mmps[mmps["mmp_year"] > cutoff]

        def aggregate_sali(df, min_mmps=3, min_pos=3):
            pos = (
                df.groupby(["core_smiles", "target_chembl_id"])
                .agg(sensitivity=("sali", "mean"), n_mmps=("sali", "count"))
                .reset_index()
            )
            pos = pos[pos["n_mmps"] >= min_mmps].reset_index(drop=True)
            mol_pos = (
                df[["mol_from", "core_smiles", "target_chembl_id"]]
                .drop_duplicates()
                .merge(pos[["core_smiles", "target_chembl_id", "sensitivity"]],
                       on=["core_smiles", "target_chembl_id"])
            )
            pos_count = (
                mol_pos.groupby(["mol_from", "target_chembl_id"]).size()
                .reset_index(name="n_pos")
            )
            pos_count = pos_count[pos_count["n_pos"] >= min_pos]
            return mol_pos.merge(
                pos_count[["mol_from", "target_chembl_id"]],
                on=["mol_from", "target_chembl_id"],
            )

        train_pos = aggregate_sali(mmps_train)
        test_pos = aggregate_sali(mmps_test)
        logger.info("    Train positions: %s, Test positions: %s",
                     f"{len(train_pos):,}", f"{len(test_pos):,}")

        if len(test_pos) < 100:
            logger.warning("    Too few test positions, skipping cutoff %d", cutoff)
            continue

        def build_features(df):
            X_list = []
            for smi in df["core_smiles"]:
                ctx = ctx_lookup.get(smi, zero_ctx)
                nh = _core_n_heavy(smi)
                nr = _core_n_rings(smi)
                X_list.append(np.append(ctx, [nh, nr]))
            return np.array(X_list, dtype=np.float32)

        X_train = build_features(train_pos)
        y_train = train_pos["sensitivity"].values.astype(np.float32)
        X_test = build_features(test_pos)
        y_test = test_pos["sensitivity"].values.astype(np.float32)
        g_test = test_pos["mol_from"].values

        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m_hgb = eval_metrics_grouped(preds, y_test, g_test)

        idx_nh = len(ctx_cols)
        m_heur = eval_metrics_grouped(-X_test[:, idx_nh], y_test, g_test)

        logger.info("    HGB=%.4f  Heur=%.4f", m_hgb["ndcg"], m_heur["ndcg"])

        results[f"cutoff_{cutoff}"] = {
            "cutoff_year": cutoff,
            "n_train_mmps": len(mmps_train),
            "n_test_mmps": len(mmps_test),
            "hgb": {"ndcg": m_hgb["ndcg"], "hit_rate": m_hgb["hit_rate"], "spearman": m_hgb["spearman"]},
            "heuristic": {"ndcg": m_heur["ndcg"], "hit_rate": m_heur["hit_rate"], "spearman": m_heur["spearman"]},
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl-sqlite", type=str, default=DEFAULT_DB)
    args = parser.parse_args()

    t0 = time.perf_counter()

    # Phase 1: Build SALI position data
    X, feat_names, groups, offsets, targets, y_sali, core_smiles = \
        build_sali_position_data()

    # Evaluation 1: Family holdout
    family_results = eval_family_holdout(X, feat_names, groups, offsets, targets, y_sali)
    with open(OUTPUT_DIR / "sali_family_holdout_results.json", "w") as f:
        json.dump(family_results, f, indent=2)
    logger.info("Saved sali_family_holdout_results.json")

    # Evaluation 2: Novel scaffold
    scaffold_results = eval_novel_scaffold(X, feat_names, groups, offsets, targets, y_sali, core_smiles)
    with open(OUTPUT_DIR / "sali_novel_scaffold_results.json", "w") as f:
        json.dump(scaffold_results, f, indent=2)
    logger.info("Saved sali_novel_scaffold_results.json")

    # Evaluation 3: Temporal split
    temporal_results = eval_temporal_split(args.chembl_sqlite)
    if temporal_results:
        with open(OUTPUT_DIR / "sali_temporal_split_results.json", "w") as f:
            json.dump(temporal_results, f, indent=2)
        logger.info("Saved sali_temporal_split_results.json")

    elapsed = time.perf_counter() - t0
    logger.info("\nAll SALI OOD evaluations complete in %.0fs", elapsed)


if __name__ == "__main__":
    main()
