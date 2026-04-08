from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from activity_cliffs.data.chembl import fetch_target_activities, resolve_chembl_sqlite_path
from activity_cliffs.data.curation import curate_chembl_activities
from activity_cliffs.data.mmp import extract_mmps

app = typer.Typer(add_completion=False, help="Extract matched molecular pairs (MMPs) from ChEMBL.")


@app.command()
def main(
    targets: List[str] = typer.Option(..., "--targets", help="One or more ChEMBL target_chembl_id values."),
    chembl_sqlite: Optional[Path] = typer.Option(None, "--chembl-sqlite", help="Path to chembl_XX.sqlite."),
    standard_type: str = typer.Option("IC50", "--standard-type", help="ChEMBL standard_type."),
    outdir: Path = typer.Option(Path("outputs/mmps"), "--outdir", help="Output directory."),
    max_group_size: int = typer.Option(200, "--max-group-size", help="Max molecules per core (prevents O(n²) explosion)."),
    min_confidence: int = typer.Option(7, "--min-confidence", help="ChEMBL assay confidence score threshold."),
):
    sqlite_path = resolve_chembl_sqlite_path(chembl_sqlite)
    outdir.mkdir(parents=True, exist_ok=True)

    all_mmps: list[pd.DataFrame] = []

    for target in targets:
        typer.echo(f"\n[{target}] Fetching activities from ChEMBL...")
        raw = fetch_target_activities(
            sqlite_path,
            target_chembl_ids=[target],
            standard_type=standard_type,
            min_confidence_score=min_confidence,
        )
        curated = curate_chembl_activities(raw).df
        typer.echo(f"[{target}] Curated molecules: {len(curated):,}")

        typer.echo(f"[{target}] Extracting MMPs (max_group_size={max_group_size})...")
        mmps = extract_mmps(curated, target_chembl_id=target, max_group_size=max_group_size)

        if len(mmps) == 0:
            typer.echo(f"[{target}] No MMPs found.")
            continue

        typer.echo(f"[{target}] MMPs found:            {len(mmps):>10,}")

        n_cliff = (mmps["abs_delta_pActivity"] >= 1.5).sum()
        typer.echo(
            f"[{target}] Cliff MMPs (|dp| >= 1.5): {n_cliff:>10,}  "
            f"({100 * n_cliff / len(mmps):.1f}%)"
        )
        typer.echo(f"[{target}] Unique cores:           {mmps['core_smiles'].nunique():>10,}")
        typer.echo(f"[{target}] Unique transforms:      {mmps['transform_smarts'].nunique():>10,}")

        # Sample transform_smarts for a sanity check
        sample = mmps[mmps["abs_delta_pActivity"] >= 1.5]["transform_smarts"].value_counts().head(5)
        if len(sample) > 0:
            typer.echo(f"[{target}] Top cliff-causing transforms:")
            for ts, cnt in sample.items():
                typer.echo(f"  {cnt:4d}x  {ts}")

        tdir = outdir / target
        tdir.mkdir(parents=True, exist_ok=True)
        out_path = tdir / "mmps.parquet"
        mmps.to_parquet(out_path, index=False)
        typer.echo(f"[{target}] Saved to {out_path}")

        all_mmps.append(mmps)

    if len(targets) > 1 and all_mmps:
        combined = pd.concat(all_mmps, ignore_index=True)
        combined_path = outdir / "all_mmps.parquet"
        combined.to_parquet(combined_path, index=False)
        typer.echo(f"\nAll targets combined: {combined_path}  ({len(combined):,} rows)")


if __name__ == "__main__":
    app()
