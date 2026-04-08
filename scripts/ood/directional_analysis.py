#!/usr/bin/env python
"""
OOD Experiment 8: Directional Analysis

Characterizes what "sensitive" means at each position:
  - "Improvable": more modifications improve potency than degrade it
  - "Fragile": more modifications degrade potency than improve it
  - "Neutral": balanced between improvements and degradations

For high-sensitivity positions, decomposes the MMP distribution into:
  - Fraction of modifications that IMPROVE by >1 log unit
  - Fraction that DECREASE by >1 log unit
  - The improvement/degradation ratio

Also tests whether any features predict the direction bias.

Usage:
    python scripts/ood/directional_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("OOD EXPERIMENT 8: DIRECTIONAL ANALYSIS")
    print("=" * 78)

    t0 = time.perf_counter()

    # ── Load MMP data ────────────────────────────────────────────────────
    print("\nLoading MMP data...")
    mmps = pd.read_parquet(
        "outputs/mmps/all_mmps.parquet",
        columns=[
            "target_chembl_id", "mol_from", "core_smiles",
            "delta_pActivity", "abs_delta_pActivity",
        ],
    )
    print(f"  {len(mmps):,} MMPs loaded")

    # ── Aggregate directional statistics per position ────────────────────
    print("\nAggregating directional statistics per (core, target) position...")

    # Define thresholds
    CLIFF_THRESH = 1.0  # pActivity units for "large" change

    mmps["improves"] = (mmps["delta_pActivity"] > CLIFF_THRESH).astype(int)
    mmps["degrades"] = (mmps["delta_pActivity"] < -CLIFF_THRESH).astype(int)
    mmps["large_change"] = (mmps["abs_delta_pActivity"] > CLIFF_THRESH).astype(int)

    pos = (
        mmps
        .groupby(["core_smiles", "target_chembl_id"])
        .agg(
            n_mmps=("delta_pActivity", "count"),
            mean_delta=("delta_pActivity", "mean"),  # signed mean
            mean_abs_delta=("abs_delta_pActivity", "mean"),  # unsigned mean
            std_delta=("delta_pActivity", "std"),
            n_improves=("improves", "sum"),
            n_degrades=("degrades", "sum"),
            n_large=("large_change", "sum"),
            max_improvement=("delta_pActivity", "max"),
            max_degradation=("delta_pActivity", "min"),
        )
        .reset_index()
    )

    # Filter for reliability
    pos = pos[pos["n_mmps"] >= 3].reset_index(drop=True)
    print(f"  {len(pos):,} positions with >= 3 MMPs")

    # Classify positions by direction bias
    pos["frac_improves"] = pos["n_improves"] / pos["n_mmps"]
    pos["frac_degrades"] = pos["n_degrades"] / pos["n_mmps"]
    pos["frac_large"] = pos["n_large"] / pos["n_mmps"]

    # Direction ratio: >1 means more improvements than degradations
    pos["direction_ratio"] = (pos["n_improves"] + 0.5) / (pos["n_degrades"] + 0.5)
    pos["log_direction_ratio"] = np.log2(pos["direction_ratio"])

    # Classify
    pos["direction_class"] = "neutral"
    pos.loc[pos["log_direction_ratio"] > 1.0, "direction_class"] = "improvable"
    pos.loc[pos["log_direction_ratio"] < -1.0, "direction_class"] = "fragile"

    # ── Overall statistics ───────────────────────────────────────────────
    print("\n--- Overall Direction Statistics ---")
    print(f"  Total positions: {len(pos):,}")
    for cls in ["improvable", "neutral", "fragile"]:
        n = (pos["direction_class"] == cls).sum()
        pct = 100 * n / len(pos)
        print(f"  {cls:12s}: {n:>8,} ({pct:5.1f}%)")

    print(f"\n  Mean signed delta:          {pos['mean_delta'].mean():.4f}")
    print(f"  Mean abs delta (sensitivity): {pos['mean_abs_delta'].mean():.4f}")
    print(f"  Mean direction ratio:         {pos['direction_ratio'].mean():.3f}")
    print(f"  Mean log2(direction ratio):   {pos['log_direction_ratio'].mean():.4f}")

    # ── High-sensitivity positions breakdown ─────────────────────────────
    print("\n--- High-Sensitivity Positions (sensitivity > 1.0) ---")
    high_sens = pos[pos["mean_abs_delta"] > 1.0]
    print(f"  N = {len(high_sens):,} ({100*len(high_sens)/len(pos):.1f}% of all)")

    for cls in ["improvable", "neutral", "fragile"]:
        n = (high_sens["direction_class"] == cls).sum()
        pct = 100 * n / len(high_sens)
        print(f"  {cls:12s}: {n:>8,} ({pct:5.1f}%)")

    print(f"\n  Among high-sensitivity positions:")
    print(f"    Mean frac improving > 1 log: {high_sens['frac_improves'].mean():.3f}")
    print(f"    Mean frac degrading > 1 log: {high_sens['frac_degrades'].mean():.3f}")
    print(f"    Mean max improvement:        {high_sens['max_improvement'].mean():.3f}")
    print(f"    Mean max degradation:        {high_sens['max_degradation'].mean():.3f}")

    # ── Low-sensitivity positions breakdown ──────────────────────────────
    print("\n--- Low-Sensitivity Positions (sensitivity < 0.5) ---")
    low_sens = pos[pos["mean_abs_delta"] < 0.5]
    print(f"  N = {len(low_sens):,} ({100*len(low_sens)/len(pos):.1f}% of all)")

    for cls in ["improvable", "neutral", "fragile"]:
        n = (low_sens["direction_class"] == cls).sum()
        pct = 100 * n / len(low_sens)
        print(f"  {cls:12s}: {n:>8,} ({pct:5.1f}%)")

    # ── Sensitivity quartiles breakdown ──────────────────────────────────
    print("\n--- Direction by Sensitivity Quartile ---")
    pos["sens_quartile"] = pd.qcut(
        pos["mean_abs_delta"], 4,
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )

    for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        qdf = pos[pos["sens_quartile"] == q]
        imp = (qdf["direction_class"] == "improvable").mean()
        neu = (qdf["direction_class"] == "neutral").mean()
        fra = (qdf["direction_class"] == "fragile").mean()
        print(f"  {q}: improvable={100*imp:.1f}%  neutral={100*neu:.1f}%  fragile={100*fra:.1f}%  "
              f"mean_frac_improve={qdf['frac_improves'].mean():.3f}  "
              f"mean_frac_degrade={qdf['frac_degrades'].mean():.3f}")

    # ── Signed mean analysis: does mean_delta predict anything? ──────────
    print("\n--- Signed Mean Delta Distribution ---")
    print(f"  Mean of signed mean: {pos['mean_delta'].mean():.4f}")
    print(f"  Std of signed mean:  {pos['mean_delta'].std():.4f}")
    print(f"  Positions with positive mean (net improvement): "
          f"{(pos['mean_delta'] > 0).sum():,} "
          f"({100*(pos['mean_delta'] > 0).mean():.1f}%)")
    print(f"  Positions with negative mean (net degradation): "
          f"{(pos['mean_delta'] < 0).sum():,} "
          f"({100*(pos['mean_delta'] < 0).mean():.1f}%)")

    # Correlation between sensitivity (unsigned) and direction bias
    rho, p = stats.spearmanr(pos["mean_abs_delta"], pos["log_direction_ratio"])
    print(f"\n  Correlation between sensitivity and direction bias:")
    print(f"    Spearman rho = {rho:.4f}, p = {p:.2e}")
    if abs(rho) < 0.05:
        print(f"    -> No relationship: sensitivity is independent of direction")
    else:
        print(f"    -> Weak relationship: higher sensitivity tends toward "
              f"{'improvement' if rho > 0 else 'degradation'}")

    # ── Feature predictiveness of direction ──────────────────────────────
    print("\n--- Can 3D context features predict direction? ---")
    # Load context features
    ctx_df = pd.read_parquet("outputs/features/context_3d.parquet")
    pos_with_ctx = pos.merge(ctx_df, on="core_smiles", how="inner")
    print(f"  {len(pos_with_ctx):,} positions with 3D context features")

    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    for col in ctx_cols:
        rho, p = stats.spearmanr(pos_with_ctx[col], pos_with_ctx["log_direction_ratio"])
        sig = "*" if p < 0.001 else ""
        print(f"    {col:30s}  rho={rho:+.4f}  p={p:.2e} {sig}")

    # ── Key finding for framing ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("KEY FINDINGS FOR FRAMING:")
    print("=" * 78)

    overall_balanced = abs(pos["log_direction_ratio"].mean()) < 0.5
    neutral_dominant = (pos["direction_class"] == "neutral").mean() > 0.5

    if overall_balanced and neutral_dominant:
        print("  The majority of positions are NEUTRAL (balanced improvements")
        print("  and degradations). This validates the model's design:")
        print("  'sensitive' means 'potency varies a lot', not 'fragile'.")
        print("  The model correctly identifies WHERE to explore, and the")
        print("  chemist decides WHICH direction based on their knowledge.")
    else:
        print("  Direction is NOT balanced. Investigate which direction")
        print("  dominates and whether this affects the model's utility.")

    # ── Save results ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    results = {
        "n_positions": len(pos),
        "direction_counts": {
            cls: int((pos["direction_class"] == cls).sum())
            for cls in ["improvable", "neutral", "fragile"]
        },
        "direction_fractions": {
            cls: float((pos["direction_class"] == cls).mean())
            for cls in ["improvable", "neutral", "fragile"]
        },
        "high_sensitivity": {
            "n": len(high_sens),
            "direction_fractions": {
                cls: float((high_sens["direction_class"] == cls).mean())
                for cls in ["improvable", "neutral", "fragile"]
            },
            "mean_frac_improves": float(high_sens["frac_improves"].mean()),
            "mean_frac_degrades": float(high_sens["frac_degrades"].mean()),
        },
        "sensitivity_direction_correlation": {
            "spearman": float(stats.spearmanr(
                pos["mean_abs_delta"], pos["log_direction_ratio"]
            )[0]),
        },
        "mean_signed_delta": float(pos["mean_delta"].mean()),
        "frac_positive_mean": float((pos["mean_delta"] > 0).mean()),
        "elapsed_s": round(elapsed, 1),
    }

    out_path = Path("outputs/ood/directional_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
