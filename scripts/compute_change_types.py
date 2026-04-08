#!/usr/bin/env python
"""
M6b: Compute R-group property vectors for all unique R-groups in the MMP corpus.

Saves a lookup table (rgroup_smiles -> 11-dim property vector) as parquet.
The change-type vector for any transform is simply:
    delta = props[rgroup_to] - props[rgroup_from]

Usage:
    python scripts/compute_change_types.py
    python scripts/compute_change_types.py --mmp-parquet outputs/mmps/all_mmps.parquet
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from activity_cliffs.features.change_type import (
    RGROUP_PROP_NAMES,
    build_rgroup_prop_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    mmp_parquet: Path = typer.Option(
        Path("outputs/mmps/all_mmps.parquet"),
        help="Path to the MMP corpus parquet",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/features"),
        help="Output directory for rgroup_props.parquet",
    ),
) -> None:
    """Compute R-group property vectors for all unique R-groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rgroup_props.parquet"

    # ── Load unique R-groups ─────────────────────────────────────────────────
    logger.info("Loading R-group SMILES from %s ...", mmp_parquet)
    cols = pd.read_parquet(mmp_parquet, columns=["rgroup_from", "rgroup_to"])
    unique_rgs = pd.concat([cols["rgroup_from"], cols["rgroup_to"]]).unique().tolist()
    n_total_rows = len(cols)
    del cols
    logger.info("  %d unique R-groups (from %d MMP rows)", len(unique_rgs), n_total_rows)

    # ── Build property cache ─────────────────────────────────────────────────
    logger.info("Computing R-group properties ...")
    t0 = time.time()
    prop_mat, smi_to_idx = build_rgroup_prop_cache(unique_rgs)
    elapsed = time.time() - t0
    logger.info("  Done in %.1f min (%.0f rg/s)", elapsed / 60, len(unique_rgs) / elapsed)

    # ── Save as parquet ──────────────────────────────────────────────────────
    df = pd.DataFrame(prop_mat, columns=RGROUP_PROP_NAMES)
    df.insert(0, "rgroup_smiles", unique_rgs)
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("  Saved %s (%d rows, %.1f MB)", out_path, len(df), out_path.stat().st_size / 1e6)

    # ── Summary stats ────────────────────────────────────────────────────────
    logger.info("Property summary:")
    for col in RGROUP_PROP_NAMES:
        s = df[col]
        logger.info(
            "  %-18s  mean=%.3f  std=%.3f  min=%.1f  max=%.1f  nonzero=%d (%.1f%%)",
            col, s.mean(), s.std(), s.min(), s.max(),
            (s != 0).sum(), 100 * (s != 0).mean(),
        )

    # ── Compute change-type delta stats on a sample ──────────────────────────
    logger.info("Sampling 100K transforms for delta statistics ...")
    mmp_sample = pd.read_parquet(
        mmp_parquet,
        columns=["rgroup_from", "rgroup_to", "abs_delta_pActivity"],
    )
    if len(mmp_sample) > 100_000:
        mmp_sample = mmp_sample.sample(100_000, random_state=42)

    idx_from = mmp_sample["rgroup_from"].map(smi_to_idx).to_numpy(dtype=np.intp)
    idx_to = mmp_sample["rgroup_to"].map(smi_to_idx).to_numpy(dtype=np.intp)
    deltas = prop_mat[idx_to] - prop_mat[idx_from]

    logger.info("Change-type delta summary (100K sample):")
    for i, name in enumerate(RGROUP_PROP_NAMES):
        col = deltas[:, i]
        logger.info(
            "  delta_%-14s  mean=%+.3f  std=%.3f  min=%+.1f  max=%+.1f",
            name, col.mean(), col.std(), col.min(), col.max(),
        )

    # Compare cliff vs non-cliff (cliff = abs_delta_pActivity >= 1.5)
    is_cliff = mmp_sample["abs_delta_pActivity"].values >= 1.5
    n_cliff = is_cliff.sum()
    logger.info("Cliff vs non-cliff delta means (%d cliffs, %d non-cliffs):", n_cliff, (~is_cliff).sum())
    for i, name in enumerate(RGROUP_PROP_NAMES):
        cliff_mean = deltas[is_cliff, i].mean()
        noncliff_mean = deltas[~is_cliff, i].mean()
        logger.info(
            "  delta_%-14s  cliff=%+.3f  non-cliff=%+.3f  diff=%+.3f",
            name, cliff_mean, noncliff_mean, cliff_mean - noncliff_mean,
        )


if __name__ == "__main__":
    app()
