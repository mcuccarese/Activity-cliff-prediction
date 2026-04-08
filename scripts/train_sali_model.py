#!/usr/bin/env python
"""
Train SALI-normalized position sensitivity model + OOD validation.

SALI (Structure-Activity Landscape Index) normalization strips the trivial
size effect: raw |ΔpActivity| is dominated by R-group size (bigger swaps =
bigger changes). SALI = |ΔpActivity| / (max_rgroup_n_heavy + 1) isolates
positions where SMALL modifications cause DISPROPORTIONATELY LARGE activity
shifts -- the true activity cliffs.

Pipeline:
  1. Load 25M MMPs, compute R-group sizes, SALI-normalize
  2. Aggregate to position level: sali_sensitivity = mean(sali_mmp)
  3. LOO-target validation (heuristic vs ML on SALI)
  4. OOD validation: novel scaffolds (80/20 core split)
  5. OOD validation: temporal split (train ≤2015, test >2015)
  6. Train final model on all data, save for webapp

Usage:
    python scripts/train_sali_model.py
    python scripts/train_sali_model.py --chembl-sqlite "D:\\Mike project data\\..."
"""
from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────────

MMPS_PATH = Path("outputs/mmps/all_mmps.parquet")
CONTEXT_3D_PATH = Path("outputs/features/context_3d.parquet")
MODEL_DIR = Path("webapp/model")
OUTPUT_JSON = Path("outputs/ood/sali_model_validation.json")
DEFAULT_DB = r"D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}

MIN_MMPS = 3
MIN_POSITIONS = 3


# ── Utility functions ──────────────────────────────────────────────────────────

def _n_heavy(smi: str) -> int:
    if not smi or smi == "[H]":
        return 0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


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


def eval_grouped(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        n = mask.sum()
        if n < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(hit_rate_at_1(y[mask], scores[mask]))
    if len(y) > 5:
        rho = float(stats.spearmanr(scores, y)[0])
        if not np.isfinite(rho):
            rho = 0.0
    else:
        rho = 0.0
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": rho,
        "n_groups": len(ndcgs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BUILD SALI POSITION DATA
# ═══════════════════════════════════════════════════════════════════════════════

def build_sali_position_data():
    """Load MMPs, SALI-normalize, aggregate to position level, attach features."""
    logger.info("PHASE 1: Building SALI position data")

    # Load MMPs
    logger.info("  Loading MMPs from %s ...", MMPS_PATH)
    mmps = pd.read_parquet(
        MMPS_PATH,
        columns=[
            "target_chembl_id", "mol_from", "core_smiles",
            "rgroup_from", "rgroup_to", "abs_delta_pActivity",
        ],
    )
    logger.info("  Loaded %s MMPs", f"{len(mmps):,}")

    # Compute R-group heavy atom counts (cached by unique R-group)
    logger.info("  Computing R-group heavy atom counts...")
    t0 = time.perf_counter()
    all_rgroups = set(mmps["rgroup_from"].unique()) | set(mmps["rgroup_to"].unique())
    logger.info("  %d unique R-groups", len(all_rgroups))
    rg_cache = {rg: _n_heavy(rg) for rg in all_rgroups}
    logger.info("  R-group sizes computed in %.0fs", time.perf_counter() - t0)

    mmps["nheavy_from"] = mmps["rgroup_from"].map(rg_cache).astype(np.float32)
    mmps["nheavy_to"] = mmps["rgroup_to"].map(rg_cache).astype(np.float32)
    mmps["max_rg_nheavy"] = np.maximum(mmps["nheavy_from"], mmps["nheavy_to"])

    # SALI normalization: |ΔpActivity| / (max_rgroup_size + 1)
    mmps["sali"] = mmps["abs_delta_pActivity"] / (mmps["max_rg_nheavy"] + 1.0)

    logger.info("  SALI stats: mean=%.3f, median=%.3f, std=%.3f",
                mmps["sali"].mean(), mmps["sali"].median(), mmps["sali"].std())

    # Aggregate to position level: (core_smiles, target) -> mean SALI
    pos = (
        mmps
        .groupby(["core_smiles", "target_chembl_id"])
        .agg(
            sali_sensitivity=("sali", "mean"),
            raw_sensitivity=("abs_delta_pActivity", "mean"),
            n_mmps=("sali", "count"),
        )
        .reset_index()
    )
    pos = pos[pos["n_mmps"] >= MIN_MMPS].reset_index(drop=True)
    logger.info("  %s position-target pairs after filtering", f"{len(pos):,}")

    # Map mol_from -> positions
    mol_pos = (
        mmps[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
        .merge(pos, on=["core_smiles", "target_chembl_id"])
    )

    # Filter: >= MIN_POSITIONS per (mol, target)
    counts = (
        mol_pos.groupby(["mol_from", "target_chembl_id"]).size()
        .reset_index(name="n_positions")
    )
    counts = counts[counts["n_positions"] >= MIN_POSITIONS]
    mol_pos = mol_pos.merge(
        counts[["mol_from", "target_chembl_id"]],
        on=["mol_from", "target_chembl_id"],
    )
    logger.info("  %s rows after position filtering", f"{len(mol_pos):,}")

    # Load 3D context features
    logger.info("  Loading 3D context features...")
    ctx_df = pd.read_parquet(CONTEXT_3D_PATH)
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32)))
    n_ctx = len(ctx_cols)

    # Core topology
    logger.info("  Computing core topology...")
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

    zero_ctx = np.zeros(n_ctx, dtype=np.float32)
    X_list = []
    for smi in mol_pos["core_smiles"]:
        ctx = ctx_lookup.get(smi, zero_ctx)
        nh, nr = core_topo.get(smi, (0, 0))
        X_list.append(np.concatenate([ctx, [float(nh), float(nr)]]))

    X = np.array(X_list, dtype=np.float32)
    feature_names = [f"ctx_{c}" for c in ctx_cols] + ["core_n_heavy", "core_n_rings"]

    groups = mol_pos["mol_from"].values.astype(np.int64)
    y_sali = mol_pos["sali_sensitivity"].values.astype(np.float32)
    y_raw = mol_pos["raw_sensitivity"].values.astype(np.float32)
    core_smiles_arr = mol_pos["core_smiles"].values

    logger.info("  Feature matrix: %s, %d targets", X.shape, len(target_names))

    return {
        "X": X,
        "y_sali": y_sali,
        "y_raw": y_raw,
        "feature_names": feature_names,
        "groups": groups,
        "target_offsets": np.array(target_offsets),
        "target_names": np.array(target_names),
        "core_smiles": core_smiles_arr,
        "mol_pos": mol_pos,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: LOO-TARGET VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def loo_target_validation(data: dict) -> dict:
    """Leave-one-target-out validation comparing heuristic vs ML on SALI."""
    logger.info("PHASE 2: LOO-target validation on SALI target")
    X, y = data["X"], data["y_sali"]
    groups = data["groups"]
    offsets = data["target_offsets"]
    targets = data["target_names"]
    feat_names = data["feature_names"]

    IDX_N_HEAVY = feat_names.index("core_n_heavy")

    results = {}

    # Random baseline
    rand_scores = np.random.RandomState(42).randn(len(y))
    rand_m = []
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        rand_m.append(eval_grouped(rand_scores[lo:hi], y[lo:hi], groups[lo:hi]))
    results["random"] = {
        "ndcg": float(np.mean([m["ndcg"] for m in rand_m])),
        "hit_rate": float(np.mean([m["hit_rate"] for m in rand_m])),
    }
    logger.info("  Random: NDCG@3=%.4f, Hit@1=%.3f",
                results["random"]["ndcg"], results["random"]["hit_rate"])

    # Heuristic: -core_n_heavy
    heur_scores = -X[:, IDX_N_HEAVY]
    heur_m = []
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        heur_m.append(eval_grouped(heur_scores[lo:hi], y[lo:hi], groups[lo:hi]))
    results["heuristic"] = {
        "ndcg": float(np.mean([m["ndcg"] for m in heur_m])),
        "hit_rate": float(np.mean([m["hit_rate"] for m in heur_m])),
        "spearman": float(np.mean([m["spearman"] for m in heur_m])),
    }
    logger.info("  Heuristic (-core_n_heavy): NDCG@3=%.4f, Hit@1=%.3f, Spearman=%.4f",
                results["heuristic"]["ndcg"], results["heuristic"]["hit_rate"],
                results["heuristic"]["spearman"])

    # HGB LOO-target
    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[lo:hi] = False
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[lo:hi])
        m = eval_grouped(preds, y[lo:hi], groups[lo:hi])
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])
        if (i + 1) % 10 == 0:
            logger.info("    ... %d/%d targets (running NDCG=%.4f)",
                        i + 1, len(targets), np.mean(all_ndcgs))
    elapsed = time.perf_counter() - t0
    results["hgb_full"] = {
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_per_target": {str(t): float(v) for t, v in zip(targets, all_ndcgs)},
        "elapsed_s": round(elapsed, 1),
    }
    logger.info("  HGB full (11 feat): NDCG@3=%.4f, Hit@1=%.3f, Spearman=%.4f [%.0fs]",
                results["hgb_full"]["ndcg"], results["hgb_full"]["hit_rate"],
                results["hgb_full"]["spearman"], elapsed)

    results["ml_vs_heuristic"] = {
        "ndcg_delta": results["hgb_full"]["ndcg"] - results["heuristic"]["ndcg"],
        "ml_wins": results["hgb_full"]["ndcg"] > results["heuristic"]["ndcg"],
    }
    logger.info("  ML vs heuristic: delta=%.4f, ML wins=%s",
                results["ml_vs_heuristic"]["ndcg_delta"],
                results["ml_vs_heuristic"]["ml_wins"])

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: OOD VALIDATION — NOVEL SCAFFOLDS
# ═══════════════════════════════════════════════════════════════════════════════

def ood_novel_scaffolds(data: dict) -> dict:
    """80/20 scaffold split: train on 80% of cores, test on held-out 20%."""
    logger.info("PHASE 3: OOD validation — novel scaffolds (80/20 core split)")
    X, y = data["X"], data["y_sali"]
    groups = data["groups"]
    feat_names = data["feature_names"]
    core_smiles = data["core_smiles"]

    IDX_N_HEAVY = feat_names.index("core_n_heavy")

    # Split cores 80/20
    unique_cores = np.unique(core_smiles)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_cores)
    n_train_cores = int(len(unique_cores) * 0.8)
    train_cores = set(unique_cores[:n_train_cores])

    train_mask = np.array([c in train_cores for c in core_smiles])
    test_mask = ~train_mask

    # Filter test to molecules with >= MIN_POSITIONS positions in test set
    test_groups = groups[test_mask]
    test_group_counts = pd.Series(test_groups).value_counts()
    valid_groups = set(test_group_counts[test_group_counts >= MIN_POSITIONS].index)
    valid_test = test_mask & np.array([g in valid_groups for g in groups])

    logger.info("  Train: %d rows (%d cores), Test: %d rows (%d cores, %d after filtering)",
                train_mask.sum(), n_train_cores,
                test_mask.sum(), len(unique_cores) - n_train_cores,
                valid_test.sum())

    if valid_test.sum() < 100:
        logger.warning("  Too few test rows, skipping novel scaffold OOD")
        return {"skipped": True}

    # Train on seen cores
    model = HistGradientBoostingRegressor(**HGB_KWARGS)
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[valid_test])
    ml_m = eval_grouped(preds, y[valid_test], groups[valid_test])

    # Heuristic on test set
    heur_scores = -X[valid_test, IDX_N_HEAVY]
    heur_m = eval_grouped(heur_scores, y[valid_test], groups[valid_test])

    # Random on test set
    rand_scores = np.random.RandomState(42).randn(valid_test.sum())
    rand_m = eval_grouped(rand_scores, y[valid_test], groups[valid_test])

    results = {
        "n_train_cores": n_train_cores,
        "n_test_cores": int(len(unique_cores) - n_train_cores),
        "n_test_rows": int(valid_test.sum()),
        "random": {"ndcg": rand_m["ndcg"], "hit_rate": rand_m["hit_rate"]},
        "heuristic": {"ndcg": heur_m["ndcg"], "hit_rate": heur_m["hit_rate"],
                      "spearman": heur_m["spearman"]},
        "hgb_full": {"ndcg": ml_m["ndcg"], "hit_rate": ml_m["hit_rate"],
                     "spearman": ml_m["spearman"]},
        "ml_vs_heuristic": {
            "ndcg_delta": ml_m["ndcg"] - heur_m["ndcg"],
            "ml_wins": ml_m["ndcg"] > heur_m["ndcg"],
        },
    }
    logger.info("  Novel scaffolds — Random: %.4f, Heuristic: %.4f, ML: %.4f (delta: %+.4f)",
                rand_m["ndcg"], heur_m["ndcg"], ml_m["ndcg"],
                ml_m["ndcg"] - heur_m["ndcg"])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: OOD VALIDATION — TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def ood_temporal_split(data: dict, chembl_db: str) -> dict:
    """Train on ≤2015, test on >2015. Requires ChEMBL SQLite for doc years."""
    logger.info("PHASE 4: OOD validation — temporal split (≤2015 / >2015)")

    db_path = Path(chembl_db)
    if not db_path.exists():
        logger.warning("  ChEMBL SQLite not found at %s, skipping temporal split", db_path)
        return {"skipped": True, "reason": "ChEMBL SQLite not found"}

    mol_pos = data["mol_pos"]
    X, y = data["X"], data["y_sali"]
    groups = data["groups"]
    feat_names = data["feature_names"]
    IDX_N_HEAVY = feat_names.index("core_n_heavy")

    # Get document years from ChEMBL
    logger.info("  Querying ChEMBL for document years...")
    conn = sqlite3.connect(str(db_path))
    year_query = """
        SELECT DISTINCT md.molregno, d.year
        FROM activities a
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        JOIN assays ass ON a.assay_id = ass.assay_id
        JOIN docs d ON ass.doc_id = d.doc_id
        WHERE d.year IS NOT NULL
    """
    year_df = pd.read_sql_query(year_query, conn)
    conn.close()

    # Map mol_from to earliest year
    mol_year = year_df.groupby("molregno")["year"].min().to_dict()
    mol_pos_years = mol_pos["mol_from"].map(mol_year)

    cutoff = 2015
    has_year = mol_pos_years.notna()
    train_mask = (has_year & (mol_pos_years <= cutoff)).values
    test_mask = (has_year & (mol_pos_years > cutoff)).values

    # Filter test to molecules with enough positions
    test_groups = groups[test_mask]
    test_group_counts = pd.Series(test_groups).value_counts()
    valid_groups = set(test_group_counts[test_group_counts >= MIN_POSITIONS].index)
    valid_test = test_mask & np.array([g in valid_groups for g in groups])

    logger.info("  Train (≤%d): %d rows, Test (>%d): %d rows (%d after filtering)",
                cutoff, train_mask.sum(), cutoff, test_mask.sum(), valid_test.sum())

    if valid_test.sum() < 100 or train_mask.sum() < 100:
        logger.warning("  Too few rows, skipping temporal split")
        return {"skipped": True, "reason": "Too few rows after year filtering"}

    # Novel core fraction
    train_cores = set(data["core_smiles"][train_mask])
    test_cores = set(data["core_smiles"][valid_test])
    novel_frac = 1.0 - len(train_cores & test_cores) / max(len(test_cores), 1)
    logger.info("  %.1f%% of test cores are novel (unseen in training)", novel_frac * 100)

    # Train and evaluate
    model = HistGradientBoostingRegressor(**HGB_KWARGS)
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[valid_test])
    ml_m = eval_grouped(preds, y[valid_test], groups[valid_test])

    heur_scores = -X[valid_test, IDX_N_HEAVY]
    heur_m = eval_grouped(heur_scores, y[valid_test], groups[valid_test])

    rand_scores = np.random.RandomState(42).randn(valid_test.sum())
    rand_m = eval_grouped(rand_scores, y[valid_test], groups[valid_test])

    results = {
        "cutoff": cutoff,
        "n_train": int(train_mask.sum()),
        "n_test": int(valid_test.sum()),
        "novel_core_fraction": round(novel_frac, 3),
        "random": {"ndcg": rand_m["ndcg"], "hit_rate": rand_m["hit_rate"]},
        "heuristic": {"ndcg": heur_m["ndcg"], "hit_rate": heur_m["hit_rate"],
                      "spearman": heur_m["spearman"]},
        "hgb_full": {"ndcg": ml_m["ndcg"], "hit_rate": ml_m["hit_rate"],
                     "spearman": ml_m["spearman"]},
        "ml_vs_heuristic": {
            "ndcg_delta": ml_m["ndcg"] - heur_m["ndcg"],
            "ml_wins": ml_m["ndcg"] > heur_m["ndcg"],
        },
    }
    logger.info("  Temporal split — Random: %.4f, Heuristic: %.4f, ML: %.4f (delta: %+.4f)",
                rand_m["ndcg"], heur_m["ndcg"], ml_m["ndcg"],
                ml_m["ndcg"] - heur_m["ndcg"])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: TRAIN FINAL MODEL + SAVE FOR WEBAPP
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_model(data: dict, validation_results: dict) -> None:
    """Train on ALL data, save model + metadata for webapp."""
    logger.info("PHASE 5: Training final SALI model on all data")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = data["X"], data["y_sali"]
    feat_names = data["feature_names"]
    targets = data["target_names"]

    t0 = time.perf_counter()
    model = HistGradientBoostingRegressor(**HGB_KWARGS)
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    logger.info("  Trained in %.1fs on %d rows", elapsed, len(y))

    # Save model
    model_path = MODEL_DIR / "position_hgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("  Saved model to %s (%d KB)", model_path, model_path.stat().st_size // 1024)

    # Feature importances
    try:
        importances = model.feature_importances_
        imp_dict = {n: float(v) for n, v in zip(feat_names, importances)}
        logger.info("  Feature importances:")
        for name, imp in sorted(imp_dict.items(), key=lambda x: -x[1]):
            logger.info("    %30s  %.4f", name, imp)
    except AttributeError:
        imp_dict = {}

    # LOO metrics
    loo = validation_results.get("loo_target", {})

    # Save metadata
    meta = {
        "target_variable": "sali_sensitivity",
        "sali_definition": "|delta_pActivity| / (max_rgroup_n_heavy + 1)",
        "feature_names": feat_names,
        "n_training_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets_trained_on": int(len(targets)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_p25": float(np.percentile(y, 25)),
        "y_p75": float(np.percentile(y, 75)),
        "y_p90": float(np.percentile(y, 90)),
        "hyperparameters": dict(HGB_KWARGS),
        "validation_metrics": {
            "ndcg_at_3": round(loo.get("hgb_full", {}).get("ndcg", 0), 4),
            "hit_at_1": round(loo.get("hgb_full", {}).get("hit_rate", 0), 3),
            "spearman": round(loo.get("hgb_full", {}).get("spearman", 0), 4),
            "heuristic_ndcg_at_3": round(loo.get("heuristic", {}).get("ndcg", 0), 4),
            "random_ndcg_at_3": round(loo.get("random", {}).get("ndcg", 0), 4),
        },
        "feature_importances": imp_dict,
    }
    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("  Saved metadata to %s", meta_path)

    # Save SALI training distribution for percentile computation in webapp
    np.savez_compressed(
        MODEL_DIR / "sali_train_dist.npz",
        y_sali=y,
    )
    logger.info("  Saved SALI training distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl-sqlite", default=DEFAULT_DB)
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Phase 1: Build data
    data = build_sali_position_data()

    # Phase 2: LOO-target validation
    loo_results = loo_target_validation(data)

    # Phase 3: OOD — novel scaffolds
    novel_results = ood_novel_scaffolds(data)

    # Phase 4: OOD — temporal split
    temporal_results = ood_temporal_split(data, args.chembl_sqlite)

    # Collect all results
    all_results = {
        "experiment": "SALI model training + OOD validation",
        "date": time.strftime("%Y-%m-%d"),
        "sali_definition": "|delta_pActivity| / (max_rgroup_n_heavy + 1)",
        "n_positions": int(len(data["y_sali"])),
        "n_targets": int(len(data["target_names"])),
        "loo_target": loo_results,
        "ood_novel_scaffolds": novel_results,
        "ood_temporal_split": temporal_results,
    }

    # Save validation results
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved validation results to %s", OUTPUT_JSON)

    # Phase 5: Train final model
    train_final_model(data, all_results)

    total = time.perf_counter() - t_start
    logger.info("All done in %.0f minutes", total / 60)

    # Print summary
    print("\n" + "=" * 70)
    print("SALI MODEL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTarget: SALI = |ΔpActivity| / (max_rgroup_n_heavy + 1)")
    print(f"Data: {len(data['y_sali']):,} positions, {len(data['target_names'])} targets\n")

    print("LOO-target validation:")
    print(f"  Random:    NDCG@3 = {loo_results['random']['ndcg']:.4f}")
    print(f"  Heuristic: NDCG@3 = {loo_results['heuristic']['ndcg']:.4f}")
    print(f"  ML model:  NDCG@3 = {loo_results['hgb_full']['ndcg']:.4f}")
    print(f"  ML vs heuristic: {loo_results['ml_vs_heuristic']['ndcg_delta']:+.4f} "
          f"({'ML wins' if loo_results['ml_vs_heuristic']['ml_wins'] else 'Heuristic wins'})")

    if not novel_results.get("skipped"):
        print(f"\nOOD — Novel scaffolds (20% held-out cores):")
        print(f"  Random:    NDCG@3 = {novel_results['random']['ndcg']:.4f}")
        print(f"  Heuristic: NDCG@3 = {novel_results['heuristic']['ndcg']:.4f}")
        print(f"  ML model:  NDCG@3 = {novel_results['hgb_full']['ndcg']:.4f}")
        print(f"  ML vs heuristic: {novel_results['ml_vs_heuristic']['ndcg_delta']:+.4f}")

    if not temporal_results.get("skipped"):
        print(f"\nOOD — Temporal split (train ≤{temporal_results['cutoff']}, test >{temporal_results['cutoff']}):")
        print(f"  Novel core fraction: {temporal_results['novel_core_fraction']:.1%}")
        print(f"  Random:    NDCG@3 = {temporal_results['random']['ndcg']:.4f}")
        print(f"  Heuristic: NDCG@3 = {temporal_results['heuristic']['ndcg']:.4f}")
        print(f"  ML model:  NDCG@3 = {temporal_results['hgb_full']['ndcg']:.4f}")
        print(f"  ML vs heuristic: {temporal_results['ml_vs_heuristic']['ndcg_delta']:+.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
