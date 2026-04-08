#!/usr/bin/env python
"""
Demo: Full SAR Exploration Pipeline

End-to-end demo: molecule -> sensitive positions -> diverse change types
-> specific synthesizable compounds to make.

Usage:
    python scripts/demo_full_pipeline.py
    python scripts/demo_full_pipeline.py --smiles "c1ccc(NC(=O)c2ccccc2)cc1"
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import typer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from activity_cliffs.features.context_3d import CONTEXT_3D_FEATURES
from activity_cliffs.features.change_type import CHANGE_TYPE_NAMES
from activity_cliffs.recommender.hypothesis_selector import (
    HypothesisSelector, AXIS_NAMES, AXIS_LABELS,
)
from activity_cliffs.recommender.compound_generator import (
    CompoundGenerator,
)

MODEL_DIR = ROOT / "webapp" / "model"
RESULTS_DIR = ROOT / "outputs" / "ood"

app = typer.Typer(add_completion=False)


def load_models():
    """Load change-type model (position model skipped due to numpy compat)."""
    # Change-type model
    with open(MODEL_DIR / "change_type_hgb.pkl", "rb") as f:
        ct_model = pickle.load(f)
    with open(MODEL_DIR / "change_type_meta.json") as f:
        ct_meta = json.load(f)

    return ct_model, ct_meta


def fragment_molecule(smiles: str) -> list[dict]:
    """Fragment a molecule into core + R-group pairs at each cut point."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMMPA

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # Use rdMMPA fragmentation (single cut)
    frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)

    positions = []
    seen_cores = set()
    for _, frag_pair in frags:
        # frag_pair is "core.rgroup" with [*:1] markers
        parts = frag_pair.split(".")
        if len(parts) != 2:
            continue

        # Determine which part is core (larger) and which is R-group
        p0_mol = Chem.MolFromSmiles(parts[0])
        p1_mol = Chem.MolFromSmiles(parts[1])
        if p0_mol is None or p1_mol is None:
            continue

        n0 = p0_mol.GetNumHeavyAtoms()
        n1 = p1_mol.GetNumHeavyAtoms()

        if n0 >= n1:
            core, rgroup = parts[0], parts[1]
        else:
            core, rgroup = parts[1], parts[0]

        if core in seen_cores:
            continue
        seen_cores.add(core)

        positions.append({
            "core_smiles": core,
            "rgroup_smiles": rgroup,
            "core_n_heavy": sum(1 for a in Chem.MolFromSmiles(core).GetAtoms()
                               if a.GetAtomicNum() > 0),
        })

    # Sort by core size (smaller = more sensitive by heuristic)
    positions.sort(key=lambda p: p["core_n_heavy"])
    return positions


def get_context_for_core(core_smiles: str) -> np.ndarray | None:
    """Look up 3D context features for a core."""
    import pandas as pd
    ctx_path = ROOT / "outputs" / "features" / "context_3d.parquet"
    ctx_df = pd.read_parquet(ctx_path)
    match = ctx_df[ctx_df["core_smiles"] == core_smiles]
    if len(match) == 0:
        return None
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    return match[ctx_cols].values[0].astype(np.float32)


def probe_change_type_model(model, meta, context_vec):
    """Probe the change-type model at +/- 1 sigma per axis."""
    n_ctx = len(CONTEXT_3D_FEATURES)
    n_prop = len(CHANGE_TYPE_NAMES)
    sigmas = meta["delta_prop_sigmas"]

    impacts = {}
    for j, ax_name in enumerate(CHANGE_TYPE_NAMES):
        sigma = sigmas.get(ax_name, 1.0)
        if sigma < 1e-6:
            sigma = 1.0

        delta_pos = np.zeros(n_prop, dtype=np.float32)
        delta_pos[j] = sigma
        x_pos = np.concatenate([context_vec, delta_pos]).reshape(1, -1)

        delta_neg = np.zeros(n_prop, dtype=np.float32)
        delta_neg[j] = -sigma
        x_neg = np.concatenate([context_vec, delta_neg]).reshape(1, -1)

        pred = max(float(model.predict(x_pos)[0]), float(model.predict(x_neg)[0]))
        impacts[ax_name] = pred

    return impacts


@app.command()
def main(
    smiles: str = typer.Option(
        "c1ccc2c(c1)cc(NC(=O)c1cccnc1)c(=O)[nH]2",
        help="Input SMILES to analyze",
    ),
    k: int = typer.Option(3, help="Number of change types to recommend per position"),
    top_positions: int = typer.Option(3, help="Number of top positions to analyze"),
    compounds_per_axis: int = typer.Option(3, help="Compounds to generate per change axis"),
) -> None:
    """Run the full SAR exploration pipeline."""
    t0 = time.time()

    print("\n" + "=" * 80)
    print("FULL SAR EXPLORATION PIPELINE")
    print("=" * 80)
    print(f"\nInput: {smiles}")
    print(f"Settings: top {top_positions} positions x {k} change types x {compounds_per_axis} compounds")

    # Load models
    print("\nLoading models...")
    ct_model, ct_meta = load_models()

    # Load correlation matrix for hypothesis selector
    with open(RESULTS_DIR / "correlated_axes_results.json") as f:
        corr_data = json.load(f)
    corr_matrix = np.array(corr_data["correlation_matrix"]["matrix"])

    selector = HypothesisSelector(
        correlation_matrix=corr_matrix,
        axis_names=AXIS_NAMES,
        diversity_weight=0.5,
    )

    generator = CompoundGenerator(
        delta_prop_sigmas=ct_meta["delta_prop_sigmas"],
    )

    # Step 1: Fragment molecule and identify positions
    print("\n" + "-" * 80)
    print("STEP 1: Identify sensitive positions")
    print("-" * 80)

    positions = fragment_molecule(smiles)
    print(f"  Found {len(positions)} fragmentable positions")

    if not positions:
        print("  ERROR: Could not fragment this molecule. Check SMILES validity.")
        return

    # Rank by heuristic (core_n_heavy, smaller = more sensitive)
    for i, pos in enumerate(positions[:top_positions]):
        print(f"\n  Position {i+1}: core_n_heavy={pos['core_n_heavy']}")
        print(f"    Core: {pos['core_smiles'][:70]}")
        print(f"    R-group: {pos['rgroup_smiles']}")

    # Step 2: For each top position, select diverse change types
    print("\n" + "-" * 80)
    print("STEP 2: Select diverse change types per position")
    print("-" * 80)

    position_plans = []
    for i, pos in enumerate(positions[:top_positions]):
        print(f"\n  === Position {i+1} (core_n_heavy={pos['core_n_heavy']}) ===")

        # Try to get 3D context; fall back to zeros
        ctx = get_context_for_core(pos["core_smiles"])
        if ctx is None:
            print("    (No 3D context in database -- using default)")
            ctx = np.zeros(len(CONTEXT_3D_FEATURES), dtype=np.float32)

        # Probe change-type model
        impacts = probe_change_type_model(ct_model, ct_meta, ctx)

        # Select diverse axes
        plan = selector.select(impacts, k=k)

        for rec in plan.recommendations:
            print(f"    {rec.rank}. {rec.family}: {rec.axis_label} "
                  f"(impact={rec.predicted_impact:.2f}, diversity={rec.diversity_score:.2f})")

        position_plans.append((pos, plan))

    # Step 3: Generate specific compounds
    print("\n" + "-" * 80)
    print("STEP 3: Generate specific synthesizable compounds")
    print("-" * 80)

    all_compound_plans = []
    for i, (pos, hyp_plan) in enumerate(position_plans):
        print(f"\n  === Position {i+1} ===")

        for rec in hyp_plan.recommendations:
            print(f"\n  --- {rec.axis_label} ---")

            cpd_plan = generator.generate_for_axis(
                mol_smiles=smiles,
                core_smiles=pos["core_smiles"],
                target_axis=rec.axis_name,
                max_compounds=compounds_per_axis,
            )

            print(generator.format_plan(cpd_plan))
            all_compound_plans.append(cpd_plan)

    # Summary
    total_compounds = sum(len(p.compounds) for p in all_compound_plans)
    total_positions = len(position_plans)

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"\n  Input molecule: {smiles}")
    print(f"  Positions analyzed: {total_positions}")
    print(f"  Change types per position: {k}")
    print(f"  Total compounds generated: {total_compounds}")
    print(f"  Time: {time.time() - t0:.1f}s")

    if total_compounds > 0:
        print(f"\n  These {total_compounds} compounds represent the most efficient")
        print(f"  first round of SAR exploration at the {total_positions} most")
        print(f"  sensitive positions, covering {k} orthogonal hypotheses each.")
    else:
        print("\n  No compounds found in the MMP corpus for this molecule.")
        print("  This molecule may not have close analogs in ChEMBL.")
        print("  Fallback: use Lib-INVENT for generative R-group decoration.")

    print("=" * 80)


if __name__ == "__main__":
    app()
