#!/usr/bin/env python
"""
OOD Experiment 4: Temporal Split

Gold-standard prospective validation: train on older data, test on newer data.

Protocol:
  1. Join MMP data with ChEMBL docs table to get document_year per activity
  2. Define cutoff year (try 2015, 2018)
  3. Train on MMPs where both molecules have doc_year <= cutoff
  4. Test on MMPs where at least one molecule has doc_year > cutoff
  5. Re-aggregate to position level for both sets
  6. Evaluate with NDCG@3 and compare to LOO-target result

Usage:
    python scripts/ood/temporal_split.py --chembl-sqlite "D:\\Mike project data\\..."
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.stdout.reconfigure(line_buffering=True)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_DB = r"D:\Mike project data\Activity cliffs\chembl_36\chembl_36_sqlite\chembl_36.db"


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


HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl-sqlite", type=str, default=DEFAULT_DB)
    args = parser.parse_args()

    print("=" * 78)
    print("OOD EXPERIMENT 4: TEMPORAL SPLIT")
    print("=" * 78)

    t0 = time.perf_counter()

    # ── Step 1: Get document_year per molregno from ChEMBL ───────────────
    print("\nQuerying ChEMBL for document years...")
    db_path = args.chembl_sqlite
    if not Path(db_path).exists():
        print(f"  ERROR: ChEMBL database not found at {db_path}")
        print(f"  Pass --chembl-sqlite <path>")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # Get (molregno, earliest document year) via activities -> assays -> docs
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
    print(f"  {len(mol_years):,} molecules with year info")
    print(f"  Year range: {mol_years['first_year'].min()}-{mol_years['last_year'].max()}")
    print(f"  Median first_year: {mol_years['first_year'].median():.0f}")

    # ── Step 2: Load MMP data and join years ─────────────────────────────
    print("\nLoading MMP data and joining years...")
    mmps = pd.read_parquet(
        "outputs/mmps/all_mmps.parquet",
        columns=[
            "target_chembl_id", "mol_from", "mol_to", "core_smiles",
            "abs_delta_pActivity", "delta_pActivity",
        ],
    )
    print(f"  {len(mmps):,} MMPs loaded")

    # Join years for mol_from and mol_to
    mol_year_map = dict(zip(mol_years["molregno"], mol_years["first_year"]))
    mmps["year_from"] = mmps["mol_from"].map(mol_year_map)
    mmps["year_to"] = mmps["mol_to"].map(mol_year_map)
    mmps["mmp_year"] = mmps[["year_from", "year_to"]].max(axis=1)

    n_with_year = mmps["mmp_year"].notna().sum()
    n_without = mmps["mmp_year"].isna().sum()
    print(f"  MMPs with year: {n_with_year:,} ({100*n_with_year/len(mmps):.1f}%)")
    print(f"  MMPs without year: {n_without:,}")

    mmps = mmps.dropna(subset=["mmp_year"]).reset_index(drop=True)
    mmps["mmp_year"] = mmps["mmp_year"].astype(int)

    # Year distribution
    print(f"\n  MMP year distribution:")
    for y in range(2000, 2026, 5):
        n = ((mmps["mmp_year"] >= y) & (mmps["mmp_year"] < y + 5)).sum()
        print(f"    {y}-{y+4}: {n:>10,} ({100*n/len(mmps):5.1f}%)")

    # ── Step 3: Load 3D context features ─────────────────────────────────
    print("\nLoading 3D context features...")
    ctx_df = pd.read_parquet("outputs/features/context_3d.parquet")
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(
        zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32))
    )

    from rdkit import Chem

    def _core_n_heavy(smi):
        mol = Chem.MolFromSmiles(smi)
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0) if mol else 0

    def _core_n_rings(smi):
        mol = Chem.MolFromSmiles(smi)
        return mol.GetRingInfo().NumRings() if mol else 0

    # ── Step 4: Temporal split experiments ────────────────────────────────
    cutoffs = [2015, 2018]
    results = {}

    for cutoff in cutoffs:
        print(f"\n{'='*78}")
        print(f"TEMPORAL SPLIT: train <= {cutoff}, test > {cutoff}")
        print(f"{'='*78}")

        mmps_train = mmps[mmps["mmp_year"] <= cutoff]
        mmps_test = mmps[mmps["mmp_year"] > cutoff]
        print(f"  Train MMPs: {len(mmps_train):,}")
        print(f"  Test MMPs:  {len(mmps_test):,}")

        # Aggregate to position level
        def aggregate_to_positions(df, min_mmps=3, min_pos=3):
            pos = (
                df.groupby(["core_smiles", "target_chembl_id"])
                .agg(
                    sensitivity=("abs_delta_pActivity", "mean"),
                    n_mmps=("abs_delta_pActivity", "count"),
                )
                .reset_index()
            )
            pos = pos[pos["n_mmps"] >= min_mmps].reset_index(drop=True)

            mol_pos = (
                df[["mol_from", "core_smiles", "target_chembl_id"]]
                .drop_duplicates()
            )
            mol_pos = mol_pos.merge(
                pos[["core_smiles", "target_chembl_id", "sensitivity"]],
                on=["core_smiles", "target_chembl_id"],
            )

            pos_count = (
                mol_pos.groupby(["mol_from", "target_chembl_id"]).size()
                .reset_index(name="n_pos")
            )
            pos_count = pos_count[pos_count["n_pos"] >= min_pos]
            mol_pos = mol_pos.merge(
                pos_count[["mol_from", "target_chembl_id"]],
                on=["mol_from", "target_chembl_id"],
            )
            return mol_pos

        train_pos = aggregate_to_positions(mmps_train)
        test_pos = aggregate_to_positions(mmps_test)

        print(f"  Train positions: {len(train_pos):,}")
        print(f"  Test positions:  {len(test_pos):,}")

        if len(test_pos) < 100:
            print(f"  Too few test positions, skipping cutoff {cutoff}")
            continue

        # Build feature matrices
        zero_ctx = np.zeros(len(ctx_cols), dtype=np.float32)

        def build_features(df):
            X_list = []
            for smi in df["core_smiles"]:
                ctx = ctx_lookup.get(smi, zero_ctx)
                nh = _core_n_heavy(smi)
                nr = _core_n_rings(smi)
                row = np.append(ctx, [nh, nr])
                X_list.append(row)
            return np.array(X_list, dtype=np.float32)

        X_train = build_features(train_pos)
        y_train = train_pos["sensitivity"].values.astype(np.float32)
        g_train = train_pos["mol_from"].values

        X_test = build_features(test_pos)
        y_test = test_pos["sensitivity"].values.astype(np.float32)
        g_test = test_pos["mol_from"].values

        print(f"  Feature shapes: train={X_train.shape}, test={X_test.shape}")

        # Train and evaluate
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m_hgb = eval_metrics_grouped(preds, y_test, g_test)
        rho_global, _ = stats.spearmanr(preds, y_test)

        print(f"\n  HGB results on future data:")
        print(f"    NDCG@3:   {m_hgb['ndcg']:.4f}")
        print(f"    Hit@1:    {m_hgb['hit_rate']:.3f}")
        print(f"    Spearman: {rho_global:.4f}")
        print(f"    N groups: {m_hgb['n_groups']}")

        # Heuristic baseline
        idx_nh = len(ctx_cols)  # core_n_heavy is after ctx features
        heur_scores = -X_test[:, idx_nh]
        m_heur = eval_metrics_grouped(heur_scores, y_test, g_test)
        rho_heur, _ = stats.spearmanr(heur_scores, y_test)

        print(f"\n  -core_n_heavy heuristic on future data:")
        print(f"    NDCG@3:   {m_heur['ndcg']:.4f}")
        print(f"    Hit@1:    {m_heur['hit_rate']:.3f}")
        print(f"    Spearman: {rho_heur:.4f}")

        # Check scaffold overlap
        train_cores = set(train_pos["core_smiles"])
        test_cores = set(test_pos["core_smiles"])
        overlap = train_cores & test_cores
        novel = test_cores - train_cores
        print(f"\n  Scaffold overlap:")
        print(f"    Train cores: {len(train_cores):,}")
        print(f"    Test cores:  {len(test_cores):,}")
        print(f"    Overlap:     {len(overlap):,} ({100*len(overlap)/len(test_cores):.1f}% of test)")
        print(f"    Novel:       {len(novel):,} ({100*len(novel)/len(test_cores):.1f}% of test)")

        results[f"cutoff_{cutoff}"] = {
            "cutoff_year": cutoff,
            "n_train_mmps": len(mmps_train),
            "n_test_mmps": len(mmps_test),
            "n_train_positions": len(train_pos),
            "n_test_positions": len(test_pos),
            "hgb": {
                "ndcg": m_hgb["ndcg"],
                "hit_rate": m_hgb["hit_rate"],
                "spearman": float(rho_global),
            },
            "heuristic": {
                "ndcg": m_heur["ndcg"],
                "hit_rate": m_heur["hit_rate"],
                "spearman": float(rho_heur),
            },
            "scaffold_overlap": {
                "n_train_cores": len(train_cores),
                "n_test_cores": len(test_cores),
                "n_overlap": len(overlap),
                "n_novel": len(novel),
                "frac_overlap": len(overlap) / len(test_cores),
            },
        }

    elapsed = time.perf_counter() - t0

    # Save
    out_path = Path("outputs/ood/temporal_split_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
