from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from activity_cliffs.data.chembl import list_top_targets, resolve_chembl_sqlite_path


app = typer.Typer(add_completion=False, help="List high-data ChEMBL targets for a given activity type.")


@app.command()
def main(
    chembl_sqlite: Optional[Path] = typer.Option(
        None, "--chembl-sqlite", exists=False, help="Path to chembl_XX.sqlite (or set CHEMBL_SQLITE_PATH)."
    ),
    standard_type: str = typer.Option("IC50", "--standard-type", help="ChEMBL standard_type (IC50, Ki, EC50, ...)."),
    min_confidence_score: int = typer.Option(7, "--min-confidence-score", help="Assay confidence_score minimum."),
    top: int = typer.Option(25, "--top", help="How many targets to show."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Optional exact organism filter (e.g., Homo sapiens)."),
):
    sqlite_path = resolve_chembl_sqlite_path(chembl_sqlite)
    targets = list_top_targets(
        sqlite_path,
        standard_type=standard_type,
        min_confidence_score=min_confidence_score,
        top_n=top,
        organism=organism,
    )

    header = f"Top {len(targets)} targets by {standard_type} (units nM, relation '=', confidence >= {min_confidence_score})"
    if organism:
        header += f", organism='{organism}'"
    print(header)
    print("-" * len(header))
    print(f"{'target_chembl_id':<16}  {'n_acts':>8}  {'n_cmpds':>9}  {'organism':<20}  pref_name")
    for t in targets:
        print(
            f"{t.target_chembl_id:<16}  {t.n_activities:>8}  {t.n_compounds:>9}  {t.organism[:20]:<20}  {t.pref_name}"
        )


if __name__ == "__main__":
    app()

