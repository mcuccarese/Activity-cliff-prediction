#!/usr/bin/env python
"""
M7a: Aggregate MMP data to position-level for SAR sensitivity prediction.

A "position" is defined by a core_smiles (scaffold with [*:1] at a specific
cut point). Position sensitivity at a target = mean |Δp| across all MMPs
with that core at that target.

Key reframe: instead of ranking specific R-group swaps (transformation level),
rank POSITIONS on a molecule by SAR sensitivity (position level). This turns
the 3D context features from dead weight (constant within transformation
groups) into the primary signal (varies between positions on the same molecule).

For evaluation, positions are grouped by (mol_from, target):
  "Among all modifiable positions on molecule X at target T,
   which is most SAR-sensitive?"

Features: 3D pharmacophore context at the attachment point (9 dims from M6a)
         + core topology features (n_heavy, n_rings).

Usage:
    python scripts/prepare_position_data.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

app = typer.Typer()


def _core_n_heavy(smi: str) -> int:
    """Count heavy atoms in a core SMILES (excluding the [*:1] dummy)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


def _core_n_rings(smi: str) -> int:
    """Count rings in a core SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return mol.GetRingInfo().NumRings()


@app.command()
def main(
    mmps_path: Path = typer.Option(
        Path("outputs/mmps/all_mmps.parquet"),
        help="Path to all_mmps.parquet",
    ),
    context_3d_path: Path = typer.Option(
        Path("outputs/features/context_3d.parquet"),
        help="Path to context_3d.parquet (M6a output)",
    ),
    output_path: Path = typer.Option(
        Path("evolve/eval_data/position_data.npz"),
        help="Output .npz file",
    ),
    min_mmps_per_position: int = typer.Option(
        3, help="Min MMPs required for a reliable sensitivity estimate",
    ),
    min_positions_per_mol: int = typer.Option(
        3, help="Min positions required per (mol_from, target) for ranking",
    ),
) -> None:
    """Aggregate MMP data to position level and build eval data."""
    t_start = time.perf_counter()

    # ── Step 1: Load MMP data (only needed columns) ──────────────────────
    logger.info("Loading MMP data from %s ...", mmps_path)
    mmps = pd.read_parquet(
        mmps_path,
        columns=[
            "target_chembl_id", "mol_from", "core_smiles",
            "abs_delta_pActivity",
        ],
    )
    logger.info(
        "  %s rows, %d targets, %d unique cores",
        f"{len(mmps):,}", mmps["target_chembl_id"].nunique(),
        mmps["core_smiles"].nunique(),
    )

    # ── Step 2: Aggregate to (core_smiles, target) → sensitivity ─────────
    logger.info("Aggregating to (core_smiles, target) positions ...")
    mmps["is_cliff"] = (mmps["abs_delta_pActivity"] > 1.5).astype(np.float32)

    pos = (
        mmps
        .groupby(["core_smiles", "target_chembl_id"])
        .agg(
            sensitivity_mean=("abs_delta_pActivity", "mean"),
            sensitivity_max=("abs_delta_pActivity", "max"),
            n_mmps=("abs_delta_pActivity", "count"),
            cliff_rate=("is_cliff", "mean"),
        )
        .reset_index()
    )
    logger.info("  %s position-target pairs before filtering", f"{len(pos):,}")

    pos = pos[pos["n_mmps"] >= min_mmps_per_position].reset_index(drop=True)
    logger.info(
        "  %s position-target pairs after filtering (n_mmps >= %d)",
        f"{len(pos):,}", min_mmps_per_position,
    )

    # ── Step 3: Map mol_from to positions ────────────────────────────────
    logger.info("Mapping mol_from to positions ...")
    mol_pos = (
        mmps[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
    )
    del mmps  # free ~2 GB

    # Join position-level sensitivity
    mol_pos = mol_pos.merge(pos, on=["core_smiles", "target_chembl_id"])
    logger.info(
        "  %s (mol_from, position, target) rows after sensitivity join",
        f"{len(mol_pos):,}",
    )

    # Keep only molecules with >= min_positions positions at each target
    pos_count = (
        mol_pos
        .groupby(["mol_from", "target_chembl_id"])
        .size()
        .reset_index(name="n_positions")
    )
    pos_count = pos_count[pos_count["n_positions"] >= min_positions_per_mol]
    mol_pos = mol_pos.merge(
        pos_count[["mol_from", "target_chembl_id"]],
        on=["mol_from", "target_chembl_id"],
    )
    logger.info(
        "  %s rows after filtering (>= %d positions per mol-target)",
        f"{len(mol_pos):,}", min_positions_per_mol,
    )

    if len(mol_pos) == 0:
        logger.error("No data remaining after filtering! Try lower thresholds.")
        raise typer.Exit(1)

    # ── Step 4: Load 3D context features ─────────────────────────────────
    logger.info("Loading 3D context features from %s ...", context_3d_path)
    ctx_df = pd.read_parquet(context_3d_path)
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(
        zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32))
    )
    n_ctx = len(ctx_cols)
    logger.info("  %d cores in lookup, %d features: %s",
                len(ctx_lookup), n_ctx, ctx_cols)

    # ── Step 5: Compute core topology features ───────────────────────────
    logger.info("Computing core topology features ...")
    unique_cores = mol_pos["core_smiles"].unique()
    core_topo: dict[str, tuple[int, int]] = {}
    for smi in unique_cores:
        core_topo[smi] = (_core_n_heavy(smi), _core_n_rings(smi))
    logger.info("  %d unique cores processed", len(core_topo))

    # ── Step 6: Build feature matrix and sort for eval ───────────────────
    logger.info("Building feature matrix ...")

    # Sort by target, then mol_from within each target
    mol_pos = mol_pos.sort_values(
        ["target_chembl_id", "mol_from", "core_smiles"]
    ).reset_index(drop=True)

    # Build target offsets
    target_names: list[str] = []
    target_offsets: list[int] = [0]
    for target, tdf in mol_pos.groupby("target_chembl_id", sort=True):
        target_names.append(str(target))
        target_offsets.append(target_offsets[-1] + len(tdf))

    # Build feature arrays
    zero_ctx = np.zeros(n_ctx, dtype=np.float32)
    X_ctx_list: list[np.ndarray] = []
    X_topo_list: list[list[float]] = []
    n_miss = 0

    for smi in mol_pos["core_smiles"]:
        # 3D context
        ctx = ctx_lookup.get(smi)
        if ctx is not None:
            X_ctx_list.append(ctx)
        else:
            X_ctx_list.append(zero_ctx)
            n_miss += 1
        # Core topology
        nh, nr = core_topo.get(smi, (0, 0))
        X_topo_list.append([float(nh), float(nr)])

    X_context = np.array(X_ctx_list, dtype=np.float32)
    X_topo = np.array(X_topo_list, dtype=np.float32)
    X_all = np.column_stack([X_context, X_topo])

    feature_names = [f"ctx_{c}" for c in ctx_cols] + [
        "core_n_heavy", "core_n_rings",
    ]

    y = mol_pos["sensitivity_mean"].values.astype(np.float32)
    groups = mol_pos["mol_from"].values.astype(np.int64)

    logger.info("  Feature matrix: %s (%d features)", X_all.shape,
                len(feature_names))
    logger.info("  3D context: %d missing -> zero-filled", n_miss)
    logger.info("  Targets: %d, unique mol_from: %d",
                len(target_names), len(np.unique(groups)))

    # ── Step 7: Save ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_all,
        y=y,
        groups=groups,
        target_offsets=np.array(target_offsets, dtype=np.int64),
        target_names=np.array(target_names),
        feature_names=np.array(feature_names),
        # Auxiliary targets for analysis
        y_max=mol_pos["sensitivity_max"].values.astype(np.float32),
        y_cliff_rate=mol_pos["cliff_rate"].values.astype(np.float32),
        n_mmps=mol_pos["n_mmps"].values.astype(np.int32),
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    elapsed = time.perf_counter() - t_start
    logger.info(
        "Saved %s (%.1f MB) in %.0f s", output_path, size_mb, elapsed,
    )

    # ── Diagnostics ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("POSITION-LEVEL DATA DIAGNOSTICS")
    print("=" * 72)
    print(f"  Total rows:        {len(mol_pos):,}")
    print(f"  Targets:           {len(target_names)}")
    print(f"  Unique cores:      {len(unique_cores):,}")
    print(f"  Unique mol_from:   {mol_pos['mol_from'].nunique():,}")
    print(f"  Features:          {len(feature_names)}")
    print()

    # Sensitivity distribution
    print("  Sensitivity (mean |delta_p|):")
    print(f"    mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f}")
    print(f"    median={np.median(y):.3f}  "
          f"p25={np.percentile(y, 25):.3f}  p75={np.percentile(y, 75):.3f}")
    print()

    # Within-molecule variance (critical diagnostic)
    print("  Within-molecule sensitivity variance (key metric):")
    within_vars = []
    within_ranges = []
    for target in target_names[:5]:
        mask = mol_pos["target_chembl_id"] == target
        tdf = mol_pos[mask]
        n_mol = tdf["mol_from"].nunique()
        pos_per_mol = tdf.groupby("mol_from").size()
        grp_var = tdf.groupby("mol_from")["sensitivity_mean"].var()
        grp_range = tdf.groupby("mol_from")["sensitivity_mean"].apply(
            lambda s: s.max() - s.min()
        )
        within_vars.extend(grp_var.dropna().tolist())
        within_ranges.extend(grp_range.tolist())
        print(f"    {target}: {n_mol} mols, "
              f"{pos_per_mol.mean():.1f}+/-{pos_per_mol.std():.1f} pos/mol, "
              f"within-var={grp_var.mean():.4f}, "
              f"within-range={grp_range.mean():.3f}")
    print()

    if within_vars:
        mean_var = np.mean(within_vars)
        mean_range = np.mean(within_ranges)
        print(f"    Overall within-mol variance: {mean_var:.4f}")
        print(f"    Overall within-mol range:    {mean_range:.3f}")
        if mean_range > 0.3:
            print("    -> Good: positions vary meaningfully within molecules")
        else:
            print("    -> Warning: low within-molecule variance, "
                  "ranking may be noisy")
    print()

    # Feature-sensitivity correlations
    print("  Feature-sensitivity Pearson correlations:")
    for j, fname in enumerate(feature_names):
        r = np.corrcoef(X_all[:, j], y)[0, 1]
        print(f"    {fname:30s}  r = {r:+.4f}")

    print("=" * 72)


if __name__ == "__main__":
    app()
