#!/usr/bin/env python
"""
Build an evidence index for the SAR Sensitivity Explorer webapp.

For each unique core_smiles in the MMP dataset:
  1. Pre-select the top evidence MMPs (most dramatic potency changes)
  2. Store the 11-feature vector (3D context + core topology)

At runtime, a query position's features are matched to the nearest cores
via k-nearest-neighbors, and the pre-stored evidence examples are displayed.

This gives the user a mechanistic explanation: "positions with similar
pharmacophore context historically showed these potency changes."

Usage:
    python scripts/build_evidence_index.py
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rdkit import Chem
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

app = typer.Typer()


def _core_n_heavy(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


def _core_n_rings(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return mol.GetRingInfo().NumRings()


# Human-readable target names (subset)
TARGET_NAMES = {
    "CHEMBL203": "EGFR",
    "CHEMBL206": "ERα",
    "CHEMBL210": "COX-1",
    "CHEMBL211": "COX-2",
    "CHEMBL214": "5-HT1A",
    "CHEMBL217": "DRD2",
    "CHEMBL218": "DRD3",
    "CHEMBL220": "ACE",
    "CHEMBL222": "PPARγ",
    "CHEMBL223": "5-HT2C",
    "CHEMBL224": "5-HT2A",
    "CHEMBL225": "HSD1",
    "CHEMBL226": "A1R",
    "CHEMBL228": "TXA2R",
    "CHEMBL230": "AR",
    "CHEMBL231": "H3R",
    "CHEMBL233": "mOR",
    "CHEMBL234": "DORA",
    "CHEMBL235": "GR",
    "CHEMBL236": "MR",
    "CHEMBL237": "kOR",
    "CHEMBL238": "PDE4",
    "CHEMBL240": "A2AR",
    "CHEMBL244": "PI3Kα",
    "CHEMBL251": "AChE",
    "CHEMBL252": "PDE5",
    "CHEMBL253": "mAChR M1",
    "CHEMBL254": "Thrombin",
    "CHEMBL256": "mAChR M2",
    "CHEMBL259": "CB2",
    "CHEMBL260": "CB1",
    "CHEMBL261": "5-HT7",
    "CHEMBL262": "DAT",
    "CHEMBL264": "H1R",
    "CHEMBL267": "SERT",
    "CHEMBL268": "NET",
    "CHEMBL270": "D4R",
    "CHEMBL276": "PKC",
    "CHEMBL279": "VEGFR2",
    "CHEMBL284": "Factor Xa",
    "CHEMBL301": "ABL1",
    "CHEMBL325": "HDAC1",
    "CHEMBL333": "mTOR",
    "CHEMBL340": "CDK2",
    "CHEMBL344": "p38α",
    "CHEMBL4005": "PI3Kδ",
    "CHEMBL4105": "PI3Kγ",
    "CHEMBL4722": "BACE1",
    "CHEMBL5145": "Aurora A",
}


@app.command()
def main(
    mmps_path: Path = typer.Option(
        Path("outputs/mmps/all_mmps.parquet"),
        help="Path to all_mmps.parquet",
    ),
    context_3d_path: Path = typer.Option(
        Path("outputs/features/context_3d.parquet"),
        help="Path to context_3d.parquet",
    ),
    output_path: Path = typer.Option(
        Path("webapp/model/evidence_index.pkl"),
        help="Output pickle file",
    ),
    top_k_per_core: int = typer.Option(
        5, help="Max evidence MMPs to store per core",
    ),
    chembl_sqlite: Path = typer.Option(
        None,
        help="Path to ChEMBL SQLite DB for compound ChEMBL ID lookup",
    ),
) -> None:
    """Build evidence index: top MMPs per core + nearest-neighbor model."""
    import sqlite3
    t_start = time.perf_counter()

    # ── Step 0: Build molregno → compound ChEMBL ID map (optional) ───────
    molregno_to_chembl: dict[int, str] = {}
    if chembl_sqlite is not None and Path(chembl_sqlite).exists():
        logger.info("Building molregno→chembl_id map from %s ...", chembl_sqlite)
        conn = sqlite3.connect(chembl_sqlite)
        try:
            rows = conn.execute(
                "SELECT molregno, chembl_id FROM molecule_dictionary"
            ).fetchall()
            molregno_to_chembl = {r[0]: r[1] for r in rows}
        finally:
            conn.close()
        logger.info("  %d compound IDs loaded", len(molregno_to_chembl))
    else:
        logger.info("No ChEMBL SQLite provided — compound IDs will be empty")

    # ── Step 1: Load 3D context features ─────────────────────────────────
    logger.info("Loading 3D context from %s ...", context_3d_path)
    ctx_df = pd.read_parquet(context_3d_path)
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    logger.info("  %d cores, %d features", len(ctx_df), len(ctx_cols))

    # ── Step 2: Compute core topology features ───────────────────────────
    logger.info("Computing core topology features ...")
    ctx_df["core_n_heavy"] = ctx_df["core_smiles"].apply(_core_n_heavy)
    ctx_df["core_n_rings"] = ctx_df["core_smiles"].apply(_core_n_rings)

    feature_cols = [f"ctx_{c}" if not c.startswith("ctx_") else c for c in ctx_cols]
    # The context_3d.parquet columns don't have ctx_ prefix, but we need the
    # feature vector in the same order as the model
    raw_feature_cols = ctx_cols + ["core_n_heavy", "core_n_rings"]

    # Build feature matrix (rows = unique cores)
    core_smiles_list = ctx_df["core_smiles"].values
    X_cores = ctx_df[raw_feature_cols].values.astype(np.float32)
    logger.info("  Feature matrix: %s", X_cores.shape)

    # ── Step 3: Fit scaler + BallTree for nearest-neighbor lookup ────────
    logger.info("Fitting StandardScaler + BallTree ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cores)
    tree = BallTree(X_scaled, metric="euclidean")
    logger.info("  BallTree built on %d cores", len(X_scaled))

    # ── Step 4: Extract top evidence MMPs per core ───────────────────────
    logger.info("Loading MMP data from %s (this may take a minute) ...", mmps_path)
    cols_needed = [
        "core_smiles", "target_chembl_id",
        "rgroup_from", "rgroup_to",
        "delta_pActivity", "abs_delta_pActivity",
        "smiles_from", "smiles_to",
        "mol_from", "mol_to",
    ]
    mmps = pd.read_parquet(mmps_path, columns=cols_needed)
    logger.info("  %s rows loaded", f"{len(mmps):,}")

    # For each core, keep top K by |delta| (most dramatic examples)
    logger.info("Selecting top %d evidence MMPs per core ...", top_k_per_core)
    mmps = mmps.sort_values("abs_delta_pActivity", ascending=False)
    evidence = (
        mmps
        .groupby("core_smiles")
        .head(top_k_per_core)
        .reset_index(drop=True)
    )
    del mmps  # free memory

    logger.info("  %s evidence rows (from %d unique cores)",
                f"{len(evidence):,}", evidence["core_smiles"].nunique())

    # Build lookup: core_smiles → list of evidence dicts
    evidence_lookup: dict[str, list[dict]] = {}
    for core_smi, grp in evidence.groupby("core_smiles"):
        examples = []
        for _, row in grp.iterrows():
            target_id = row["target_chembl_id"]
            target_name = TARGET_NAMES.get(target_id, target_id)
            mol_from = int(row["mol_from"]) if pd.notna(row["mol_from"]) else None
            mol_to = int(row["mol_to"]) if pd.notna(row["mol_to"]) else None
            examples.append({
                "target_id": target_id,
                "target_name": target_name,
                "rgroup_from": row["rgroup_from"],
                "rgroup_to": row["rgroup_to"],
                "delta_pActivity": float(row["delta_pActivity"]),
                "abs_delta": float(row["abs_delta_pActivity"]),
                "smiles_from": str(row["smiles_from"]) if pd.notna(row.get("smiles_from")) else "",
                "smiles_to": str(row["smiles_to"]) if pd.notna(row.get("smiles_to")) else "",
                "molecule_chembl_id_from": molregno_to_chembl.get(mol_from, "") if mol_from else "",
                "molecule_chembl_id_to": molregno_to_chembl.get(mol_to, "") if mol_to else "",
            })
        evidence_lookup[str(core_smi)] = examples
    del evidence

    logger.info("  Evidence lookup: %d cores with examples", len(evidence_lookup))

    # ── Step 5: Build core_smiles → index mapping ────────────────────────
    core_to_idx = {smi: i for i, smi in enumerate(core_smiles_list)}

    # ── Step 6: Save everything ──────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = {
        "scaler": scaler,
        "tree": tree,
        "core_smiles": core_smiles_list,  # array of core SMILES, indexed same as tree
        "core_to_idx": core_to_idx,
        "X_cores": X_cores,               # raw features for reference
        "evidence_lookup": evidence_lookup,
        "feature_cols": raw_feature_cols,
        "target_names": TARGET_NAMES,
    }

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / 1024 / 1024
    elapsed = time.perf_counter() - t_start
    logger.info("Saved %s (%.1f MB) in %.0f s", output_path, size_mb, elapsed)

    # ── Diagnostics ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EVIDENCE INDEX DIAGNOSTICS")
    print("=" * 72)
    print(f"  Cores in index:      {len(core_smiles_list):,}")
    print(f"  Cores with evidence: {len(evidence_lookup):,}")
    print(f"  Features:            {len(raw_feature_cols)}")
    print(f"  File size:           {size_mb:.1f} MB")

    # Coverage check
    n_with = sum(1 for s in core_smiles_list if s in evidence_lookup)
    print(f"  Coverage:            {n_with}/{len(core_smiles_list)} "
          f"({100*n_with/len(core_smiles_list):.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    app()
