from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from sklearn.exceptions import ConvergenceWarning
import warnings

from activity_cliffs.models import train_baselines_pairwise


app = typer.Typer(add_completion=False, help="Train baseline cliff models on mined cliff pairs.")


@app.command()
def main(
    series_parquet: Path = typer.Option(..., "--series", exists=True, help="Path to chembl_series_<target>_<type>.parquet"),
    pairs_parquet: Path = typer.Option(..., "--pairs", exists=True, help="Path to cliff_pairs_<target>_<type>.parquet"),
    outdir: Path = typer.Option(Path("outputs"), "--outdir", help="Output directory."),
):
    outdir.mkdir(parents=True, exist_ok=True)

    df_series = pd.read_parquet(series_parquet)
    df_pairs = pd.read_parquet(pairs_parquet)
    if len(df_pairs) == 0:
        raise ValueError("Pairs parquet is empty. Mine cliffs first or loosen thresholds.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        artifacts = train_baselines_pairwise(df_series, df_pairs)

    metrics_path = outdir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(artifacts.metrics, indent=2))
    print(f"Wrote: {metrics_path}")
    print(json.dumps(artifacts.metrics, indent=2))


if __name__ == "__main__":
    app()

