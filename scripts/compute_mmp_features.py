"""
Compute MMP features from all_mmps.parquet  (M4 step).

Usage
-----
python scripts/compute_mmp_features.py \\
    --input  outputs/mmps/all_mmps.parquet \\
    --output outputs/features/mmp_features.parquet

After the run finishes, EGFR (CHEMBL203) summary stats are printed.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer

from activity_cliffs.features.mmp_features import build_mmp_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Compute MMP features (M4).")


def _validate_egfr(
    mmps_parquet: Path,
    features_parquet: Path,
    target_id: str = "CHEMBL203",
) -> None:
    """Print feature summary statistics for the EGFR (CHEMBL203) subset."""
    typer.echo(f"\n=== EGFR validation ({target_id}) ===")

    # Find EGFR row positions in the MMP parquet
    target_col = pd.read_parquet(mmps_parquet, columns=["target_chembl_id"])
    mask = (target_col["target_chembl_id"] == target_id).values
    n_egfr = int(mask.sum())
    typer.echo(f"  EGFR rows: {n_egfr:,}")

    if n_egfr == 0:
        typer.echo("  No EGFR rows found — skipping validation.")
        return

    # Read feature parquet in one pass (11 columns, float32 + binary + uint32)
    feat_df = pd.read_parquet(features_parquet, engine="pyarrow")
    egfr_feats = feat_df.loc[mask]

    # ── Delta descriptor stats ────────────────────────────────────────────────
    delta_cols = [
        "delta_MW", "delta_LogP", "delta_TPSA",
        "delta_HBDonors", "delta_HBAcceptors",
        "delta_RotBonds", "delta_HeavyAtomCount",
    ]
    typer.echo("\n  Delta-descriptor summary (EGFR):")
    stats = egfr_feats[delta_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    typer.echo(stats.to_string())

    # ── FP non-zero rate (sanity check that FPs were computed) ────────────────
    def _fp_nonzero_rate(col: pd.Series) -> float:
        """Fraction of rows where the packed FP is not all-zero bytes."""
        zero_fp = bytes(32)
        return float((col != zero_fp).mean())

    rate_from = _fp_nonzero_rate(egfr_feats["fp_rgroup_from"])
    rate_to   = _fp_nonzero_rate(egfr_feats["fp_rgroup_to"])
    typer.echo(f"\n  FP non-zero rate — from: {rate_from:.3f}  to: {rate_to:.3f}")

    # ── Env hash diversity (number of unique hashes) ──────────────────────────
    n_unique_r1 = egfr_feats["env_hash_r1"].nunique()
    n_unique_r2 = egfr_feats["env_hash_r2"].nunique()
    typer.echo(
        f"  Attachment env unique hashes — r=1: {n_unique_r1:,}  r=2: {n_unique_r2:,}"
    )

    # ── Cliff vs non-cliff delta_MW distribution (quick sanity check) ─────────
    delta_pa = pd.read_parquet(mmps_parquet, columns=["abs_delta_pActivity"]).loc[mask]
    is_cliff = delta_pa["abs_delta_pActivity"] >= 1.5
    typer.echo(
        f"\n  Mean |delta_MW| — cliff: "
        f"{egfr_feats.loc[is_cliff.values, 'delta_MW'].abs().mean():.2f}  "
        f"non-cliff: {egfr_feats.loc[~is_cliff.values, 'delta_MW'].abs().mean():.2f}"
    )
    typer.echo(
        f"  Mean |delta_HeavyAtomCount| — cliff: "
        f"{egfr_feats.loc[is_cliff.values, 'delta_HeavyAtomCount'].abs().mean():.2f}  "
        f"non-cliff: {egfr_feats.loc[~is_cliff.values, 'delta_HeavyAtomCount'].abs().mean():.2f}"
    )
    typer.echo("")


@app.command()
def main(
    input_parquet: Path = typer.Option(
        Path("outputs/mmps/all_mmps.parquet"),
        "--input",
        help="Path to all_mmps.parquet.",
    ),
    output_parquet: Path = typer.Option(
        Path("outputs/features/mmp_features.parquet"),
        "--output",
        help="Destination path for mmp_features.parquet.",
    ),
    chunk_size: int = typer.Option(
        500_000,
        "--chunk-size",
        help="Rows per processing batch (500 K ≈ 100–150 MB RAM per chunk).",
    ),
    skip_if_exists: bool = typer.Option(
        False,
        "--skip-if-exists",
        help="Skip recomputation if output already exists.",
    ),
) -> None:
    if skip_if_exists and output_parquet.exists():
        typer.echo(f"Output already exists, skipping: {output_parquet}")
    else:
        t0 = time.time()
        build_mmp_features(input_parquet, output_parquet, chunk_size=chunk_size)
        elapsed = time.time() - t0
        typer.echo(f"\nFeature computation complete — {elapsed / 60:.1f} min")

        # Print output file size
        size_mb = output_parquet.stat().st_size / 1e6
        pf = pq.ParquetFile(output_parquet)
        typer.echo(f"Output: {output_parquet}  ({size_mb:.0f} MB, {pf.metadata.num_rows:,} rows)")

    _validate_egfr(input_parquet, output_parquet)


if __name__ == "__main__":
    app()
