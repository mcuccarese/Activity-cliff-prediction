"""
Compute 3D pharmacophore context features for MMP attachment points (M6a).

Usage
-----
# Test on 100 EGFR cores first:
python scripts/compute_3d_context.py --test-egfr 100

# Full run on all ~104K unique cores:
python scripts/compute_3d_context.py

# Custom I/O:
python scripts/compute_3d_context.py \
    --input  outputs/mmps/all_mmps.parquet \
    --output outputs/features/context_3d.parquet
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from activity_cliffs.features.context_3d import (
    CONTEXT_3D_FEATURES,
    build_context_3d_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Compute 3D context features (M6a).")


def _save_context_table(
    context_mat: np.ndarray,
    smi_to_idx: dict[str, int],
    output_path: Path,
) -> None:
    """Save context features as a parquet lookup table (core_smiles → features)."""
    # Invert the dict to get ordered list of SMILES
    idx_to_smi = {v: k for k, v in smi_to_idx.items()}
    smiles_list = [idx_to_smi[i] for i in range(len(idx_to_smi))]

    columns = {"core_smiles": smiles_list}
    for j, name in enumerate(CONTEXT_3D_FEATURES):
        columns[name] = context_mat[:, j].tolist()

    df = pd.DataFrame(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Saved %d cores → %s", len(df), output_path)


def _print_summary(context_mat: np.ndarray, label: str = "") -> None:
    """Print descriptive statistics of computed features."""
    if label:
        typer.echo(f"\n=== {label} ===")

    df = pd.DataFrame(context_mat, columns=CONTEXT_3D_FEATURES)
    typer.echo(df.describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())

    n_all_zero = int(np.all(context_mat == 0, axis=1).sum())
    typer.echo(f"\n  All-zero rows (parse failure): {n_all_zero}")

    # Embedding success: rows where at least one 3D-dependent feature is nonzero
    # (features 0-4 and 8 require 3D; 5-7 are topological)
    three_d_cols = context_mat[:, [0, 1, 2, 3, 4, 8]]
    n_has_3d = int((three_d_cols != 0).any(axis=1).sum())
    typer.echo(f"  Rows with 3D features: {n_has_3d} / {len(context_mat)}")
    typer.echo("")


@app.command()
def main(
    input_parquet: Path = typer.Option(
        Path("outputs/mmps/all_mmps.parquet"),
        "--input",
        help="Path to all_mmps.parquet.",
    ),
    output_parquet: Path = typer.Option(
        Path("outputs/features/context_3d.parquet"),
        "--output",
        help="Destination for context_3d.parquet lookup table.",
    ),
    test_egfr: int = typer.Option(
        0,
        "--test-egfr",
        help="If > 0, only process this many EGFR cores (for quick validation).",
    ),
    skip_if_exists: bool = typer.Option(
        False,
        "--skip-if-exists",
        help="Skip recomputation if output already exists.",
    ),
) -> None:
    if skip_if_exists and output_parquet.exists():
        typer.echo(f"Output already exists, skipping: {output_parquet}")
        return

    # ── Load unique core SMILES ──────────────────────────────────────────────
    typer.echo(f"Loading cores from {input_parquet} ...")
    t0 = time.time()

    if test_egfr > 0:
        # Test mode: only EGFR cores
        mmps = pd.read_parquet(
            input_parquet,
            columns=["target_chembl_id", "core_smiles"],
        )
        egfr_cores = (
            mmps.loc[mmps["target_chembl_id"] == "CHEMBL203", "core_smiles"]
            .unique()
            .tolist()
        )
        unique_cores = egfr_cores[:test_egfr]
        typer.echo(
            f"  Test mode: {len(unique_cores)} EGFR cores "
            f"(of {len(egfr_cores)} total EGFR cores)"
        )
        del mmps
    else:
        # Full run: all unique cores
        core_col = pd.read_parquet(input_parquet, columns=["core_smiles"])
        unique_cores = core_col["core_smiles"].unique().tolist()
        typer.echo(f"  {len(unique_cores):,} unique cores to process")
        del core_col

    # ── Compute 3D context features ──────────────────────────────────────────
    typer.echo("Computing 3D pharmacophore context features ...")
    context_mat, smi_to_idx = build_context_3d_cache(unique_cores)

    elapsed = time.time() - t0
    rate = len(unique_cores) / elapsed if elapsed > 0 else 0
    typer.echo(
        f"\nDone — {len(unique_cores):,} cores in {elapsed:.1f}s "
        f"({rate:.1f} cores/s)"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    _save_context_table(context_mat, smi_to_idx, output_parquet)

    size_mb = output_parquet.stat().st_size / 1e6
    typer.echo(f"Output: {output_parquet}  ({size_mb:.1f} MB)")

    # ── Summary stats ────────────────────────────────────────────────────────
    label = f"3D context features ({'EGFR test' if test_egfr > 0 else 'all cores'})"
    _print_summary(context_mat, label)


if __name__ == "__main__":
    app()
