#!/usr/bin/env python
"""
External validation of the compound selector / change-type recommendation logic.

Tests whether recommendations from a model trained on ChEMBL actually align with
SAR observed in independent datasets (OpenFE benchmark, Schrodinger FEP, COVID Moonshot).

Protocol:
  Phase 1: Extract MMPs from external data using rdMMPA
  Phase 2: Filter to novel scaffolds (not seen in ChEMBL training)
  Phase 3: Build ground-truth activity profiles per position
  Phase 4: Run the change-type model and compare to ground truth
  Phase 5: Generate results JSON and Figure 9

Usage:
    python scripts/external_validation/run_external_validation.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMMPA, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from activity_cliffs.features.change_type import (
    CHANGE_TYPE_NAMES,
    RGROUP_PROP_NAMES,
    compute_rgroup_props,
)
from activity_cliffs.features.context_3d import (
    CONTEXT_3D_FEATURES,
    compute_3d_context,
)
from activity_cliffs.recommender.hypothesis_selector import AXIS_NAMES, AXIS_LABELS

EXT_PARQUET = Path("D:/Mike project data/Activity cliffs/external_validation/combined_external.parquet")
ALL_MMPS_PATH = ROOT / "outputs" / "mmps" / "all_mmps.parquet"
RGROUP_PROPS_PATH = ROOT / "outputs" / "features" / "rgroup_props.parquet"
CT_MODEL_PATH = ROOT / "webapp" / "model" / "change_type_hgb.pkl"
CT_META_PATH = ROOT / "webapp" / "model" / "change_type_meta.json"
CORR_RESULTS_PATH = ROOT / "outputs" / "ood" / "correlated_axes_results.json"

OUTPUT_JSON = ROOT / "outputs" / "ood" / "external_validation_results.json"
OUTPUT_FIG = ROOT / "outputs" / "whitepaper" / "figure9_external_validation.png"

N_AXES = len(AXIS_NAMES)
K = 3  # number of axes to select
N_RANDOM_DRAWS = 100
MIN_MMPS_PER_POSITION = 10
MIN_DISTINCT_AXES = 3


def p(msg: str):
    """Print with flush for real-time monitoring."""
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Extract MMPs from external data
# ══════════════════════════════════════════════════════════════════════════

def _parse_rdmmpa_fragments(sidechains_smi: str) -> tuple[str, str] | None:
    """
    Parse rdMMPA single-cut output into (core_smiles, rgroup_smiles).

    rdMMPA.FragmentMol with maxCuts=1 returns ('', 'frag1.frag2') where the
    two fragments are dot-separated in one SMILES string. The larger fragment
    is the core, the smaller is the R-group.

    Returns (core_smiles, rgroup_smiles) with [*:1] attachment markers,
    or None if parsing fails.
    """
    combined = Chem.MolFromSmiles(sidechains_smi)
    if combined is None:
        return None

    frag_mols = Chem.GetMolFrags(combined, asMols=True, sanitizeFrags=True)
    if len(frag_mols) != 2:
        return None

    # Count heavy atoms (excluding dummy atoms) to determine core vs R-group
    def heavy_count(m):
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 0)

    hc0 = heavy_count(frag_mols[0])
    hc1 = heavy_count(frag_mols[1])

    if hc0 >= hc1:
        core_mol, rg_mol = frag_mols[0], frag_mols[1]
    else:
        core_mol, rg_mol = frag_mols[1], frag_mols[0]

    core_smi = Chem.MolToSmiles(core_mol)
    rg_smi = Chem.MolToSmiles(rg_mol)

    return core_smi, rg_smi


def extract_mmps_from_target(target_df: pd.DataFrame, target_id: str) -> pd.DataFrame:
    """
    Extract matched molecular pairs from a set of compounds using rdMMPA.

    For each pair of compounds sharing a common core (single-cut fragmentation),
    compute abs_delta_pActivity and return the MMP table.
    """
    # Parse all molecules
    smiles_list = target_df["smiles"].tolist()
    activity_list = target_df["activity_value"].tolist()

    mols = []
    valid_smiles = []
    valid_activities = []
    for smi, act in zip(smiles_list, activity_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and np.isfinite(act):
            mols.append(mol)
            valid_smiles.append(smi)
            valid_activities.append(act)

    if len(mols) < 2:
        return pd.DataFrame()

    # Fragment all molecules (single-cut)
    fragments_by_mol = {}  # idx -> list of (core, rgroup)
    for i, mol in enumerate(mols):
        try:
            frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
        except Exception:
            continue
        parsed = []
        for _empty_core, sidechains in frags:
            if not sidechains:
                continue
            result = _parse_rdmmpa_fragments(sidechains)
            if result is not None:
                parsed.append(result)
        if parsed:
            fragments_by_mol[i] = parsed

    # Build a core -> [(mol_idx, rgroup)] mapping
    core_to_mols = defaultdict(list)
    for mol_idx, frag_list in fragments_by_mol.items():
        for core_smi, rg_smi in frag_list:
            core_to_mols[core_smi].append((mol_idx, rg_smi))

    # Generate MMPs: pairs of molecules sharing the same core
    rows = []
    for core_smi, entries in core_to_mols.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                idx_a, rg_a = entries[i]
                idx_b, rg_b = entries[j]
                if rg_a == rg_b:
                    continue  # same R-group, no change

                act_a = valid_activities[idx_a]
                act_b = valid_activities[idx_b]
                delta = act_b - act_a
                abs_delta = abs(delta)

                rows.append({
                    "target_id": target_id,
                    "core_smiles": core_smi,
                    "smiles_from": valid_smiles[idx_a],
                    "smiles_to": valid_smiles[idx_b],
                    "rgroup_from": rg_a,
                    "rgroup_to": rg_b,
                    "pActivity_from": act_a,
                    "pActivity_to": act_b,
                    "delta_pActivity": delta,
                    "abs_delta_pActivity": abs_delta,
                })

    return pd.DataFrame(rows)


def phase1_extract_mmps(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Extract MMPs from all targets in the external dataset."""
    p("\n" + "=" * 70)
    p("PHASE 1: Extract MMPs from external data")
    p("=" * 70)

    # Filter to numeric, non-censored activity values
    mask = ext_df["activity_value"].notna() & np.isfinite(ext_df["activity_value"])
    ext_df = ext_df[mask].copy()
    p(f"  Compounds with numeric activity: {len(ext_df)}")

    all_mmps = []
    target_counts = {}

    targets = sorted(ext_df["target_id"].unique())
    p(f"  Processing {len(targets)} targets...")

    for i, target_id in enumerate(targets):
        target_df = ext_df[ext_df["target_id"] == target_id]
        mmps = extract_mmps_from_target(target_df, target_id)
        n_mmps = len(mmps)
        if n_mmps > 0:
            all_mmps.append(mmps)
            target_counts[target_id] = n_mmps
        if (i + 1) % 20 == 0 or (i + 1) == len(targets):
            p(f"    {i+1}/{len(targets)} targets processed...")

    if not all_mmps:
        p("  ERROR: No MMPs extracted from external data!")
        return pd.DataFrame()

    mmps_df = pd.concat(all_mmps, ignore_index=True)
    p(f"\n  Total MMPs extracted: {len(mmps_df):,}")
    p(f"  Targets with MMPs: {len(target_counts)}")

    # Per-target summary (top 10)
    sorted_targets = sorted(target_counts.items(), key=lambda x: -x[1])
    p(f"\n  Top targets by MMP count:")
    for tid, n in sorted_targets[:10]:
        p(f"    {tid}: {n:,} MMPs")
    if len(sorted_targets) > 10:
        p(f"    ... and {len(sorted_targets) - 10} more")

    return mmps_df


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Filter to novel scaffolds
# ══════════════════════════════════════════════════════════════════════════

def get_murcko_scaffold(smi: str) -> str | None:
    """Return canonical Murcko generic scaffold SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return None


def load_training_core_scaffolds() -> set[str]:
    """Load unique core_smiles from training data and extract their Murcko scaffolds."""
    import pyarrow.parquet as pq

    p("  Loading training core scaffolds from all_mmps.parquet...")
    pf = pq.ParquetFile(ALL_MMPS_PATH)

    training_cores = set()
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["core_smiles"])
        cores = table.column("core_smiles").to_pylist()
        training_cores.update(cores)
        if (rg_idx + 1) % 10 == 0:
            p(f"    Row group {rg_idx+1}/{pf.metadata.num_row_groups}: "
              f"{len(training_cores):,} unique cores so far")

    p(f"  Total unique training cores: {len(training_cores):,}")

    # Extract scaffolds from cores
    p("  Extracting Murcko scaffolds from training cores...")
    training_scaffolds = set()
    for i, core_smi in enumerate(training_cores):
        scaf = get_murcko_scaffold(core_smi)
        if scaf:
            training_scaffolds.add(scaf)
        if (i + 1) % 20000 == 0:
            p(f"    {i+1}/{len(training_cores)} cores processed...")

    p(f"  Training scaffolds: {len(training_scaffolds):,}")
    return training_scaffolds


def phase2_filter_novel_scaffolds(
    mmps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove MMPs whose core scaffold overlaps with ChEMBL training data."""
    p("\n" + "=" * 70)
    p("PHASE 2: Filter to novel scaffolds")
    p("=" * 70)

    n_before = len(mmps_df)

    # Get training scaffolds
    training_scaffolds = load_training_core_scaffolds()

    # Extract scaffolds from external MMP cores
    p("  Extracting scaffolds from external MMP cores...")
    unique_cores = mmps_df["core_smiles"].unique()
    p(f"  External unique cores: {len(unique_cores)}")

    core_scaffold_map = {}
    core_is_novel = {}
    for core_smi in unique_cores:
        scaf = get_murcko_scaffold(core_smi)
        core_scaffold_map[core_smi] = scaf
        if scaf is None:
            core_is_novel[core_smi] = True  # keep if can't compute scaffold
        else:
            core_is_novel[core_smi] = scaf not in training_scaffolds

    n_novel_cores = sum(core_is_novel.values())
    p(f"  Novel cores: {n_novel_cores} / {len(unique_cores)} "
      f"({100*n_novel_cores/max(len(unique_cores),1):.1f}%)")

    # Filter MMPs
    mmps_df = mmps_df[mmps_df["core_smiles"].map(core_is_novel)].copy()
    mmps_df = mmps_df.reset_index(drop=True)

    p(f"\n  MMPs before scaffold filter: {n_before:,}")
    p(f"  MMPs after scaffold filter:  {len(mmps_df):,}")
    p(f"  Percent retained: {100*len(mmps_df)/max(n_before,1):.1f}%")

    return mmps_df


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Build position profiles from external data
# ══════════════════════════════════════════════════════════════════════════

def compute_rgroup_props_cached(
    rgroup_smiles_set: set[str],
    existing_lookup: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute R-group properties, using existing lookup where available."""
    result = {}
    n_computed = 0
    n_cached = 0

    for smi in rgroup_smiles_set:
        if smi in existing_lookup:
            result[smi] = existing_lookup[smi]
            n_cached += 1
        else:
            props = compute_rgroup_props(smi)
            result[smi] = props
            n_computed += 1

    p(f"    R-group props: {n_cached} from cache, {n_computed} newly computed")
    return result


def phase3_build_profiles(
    mmps_df: pd.DataFrame,
    sigmas: dict[str, float],
    min_mmps: int | None = None,
    min_axes: int | None = None,
) -> list[dict]:
    """Build ground-truth activity profiles for each (core, target) position."""
    if min_mmps is None:
        min_mmps = MIN_MMPS_PER_POSITION
    if min_axes is None:
        min_axes = MIN_DISTINCT_AXES
    p("\n" + "=" * 70)
    p("PHASE 3: Build position profiles from external data")
    p("=" * 70)

    # Load existing rgroup_props for cache hits
    p("  Loading existing R-group property lookup...")
    rg_df = pd.read_parquet(RGROUP_PROPS_PATH)
    existing_lookup = {}
    for _, row in rg_df.iterrows():
        existing_lookup[row["rgroup_smiles"]] = np.array(
            [row[c] for c in RGROUP_PROP_NAMES], dtype=np.float32
        )
    p(f"  Existing lookup: {len(existing_lookup):,} R-groups")
    del rg_df

    # Collect all unique R-groups from external MMPs
    all_rgroups = set(mmps_df["rgroup_from"].unique()) | set(mmps_df["rgroup_to"].unique())
    p(f"  Unique R-groups in external MMPs: {len(all_rgroups)}")

    # Compute properties
    rg_props = compute_rgroup_props_cached(all_rgroups, existing_lookup)
    del existing_lookup

    # Compute delta properties for each MMP
    p("  Computing delta properties for external MMPs...")
    sigma_vec = np.array(
        [max(sigmas.get(c, 1.0), 1e-6) for c in CHANGE_TYPE_NAMES],
        dtype=np.float32,
    )

    n = len(mmps_df)
    delta_mat = np.empty((n, len(RGROUP_PROP_NAMES)), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    for i, (rg_from, rg_to) in enumerate(
        zip(mmps_df["rgroup_from"].values, mmps_df["rgroup_to"].values)
    ):
        pf = rg_props.get(rg_from)
        pt = rg_props.get(rg_to)
        if pf is None or pt is None:
            valid_mask[i] = False
            delta_mat[i] = 0
        else:
            delta_mat[i] = pt - pf

    n_dropped = (~valid_mask).sum()
    p(f"  Dropped {n_dropped:,} MMPs with missing R-group props")

    mmps_df = mmps_df[valid_mask].reset_index(drop=True)
    delta_mat = delta_mat[valid_mask]

    # Add delta columns
    for j, delta_name in enumerate(CHANGE_TYPE_NAMES):
        mmps_df[delta_name] = delta_mat[:, j]

    # Standardize and assign dominant axis
    standardized = np.abs(delta_mat) / sigma_vec
    dominant_idx = np.argmax(standardized, axis=1)
    mmps_df["dominant_axis"] = [CHANGE_TYPE_NAMES[i] for i in dominant_idx]

    p(f"  {len(mmps_df):,} MMPs with delta props and dominant axis")

    # Group by (core_smiles, target_id) and build profiles
    p("\n  Building per-position activity profiles...")
    grouped = mmps_df.groupby(["core_smiles", "target_id"])

    positions = []
    n_skipped_small = 0
    n_skipped_axes = 0

    for (core, target), group in grouped:
        if len(group) < min_mmps:
            n_skipped_small += 1
            continue

        n_distinct_axes = group["dominant_axis"].nunique()
        if n_distinct_axes < min_axes:
            n_skipped_axes += 1
            continue

        # Ground truth profile: mean |delta_pActivity| per dominant axis
        profile = {}
        for ax in CHANGE_TYPE_NAMES:
            mask = group["dominant_axis"] == ax
            if mask.sum() > 0:
                profile[ax] = float(group.loc[mask, "abs_delta_pActivity"].mean())
            else:
                profile[ax] = 0.0

        source_datasets = list(group["target_id"].str.split("_").str[0].unique())
        # Infer source_dataset from target_id prefix
        if target.startswith("moonshot"):
            source = "covid_moonshot"
        elif target.startswith("openfe"):
            source = "openfe_benchmark"
        else:
            source = "schrodinger_fep"

        positions.append({
            "core_smiles": core,
            "target_id": target,
            "source_dataset": source,
            "n_mmps": int(len(group)),
            "n_distinct_axes": int(n_distinct_axes),
            "ground_truth_profile": profile,
        })

    p(f"\n  Positions with >= {min_mmps} MMPs: "
      f"{n_skipped_small + n_skipped_axes + len(positions)}")
    p(f"  Skipped (too few MMPs): {n_skipped_small}")
    p(f"  Skipped (too few distinct axes): {n_skipped_axes}")
    p(f"  Valid positions for validation: {len(positions)}")

    return positions


# ══════════════════════════════════════════════════════════════════════════
# Phase 4: Run compound selector and compare
# ══════════════════════════════════════════════════════════════════════════

def probe_model_at_context(model, meta, context_vec):
    """Probe the change-type model along each axis at +/- 1 sigma."""
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
        impacts[ax_name] = max(0.0, pred)

    return impacts


def select_top_impact(impacts: dict[str, float], k: int) -> list[str]:
    """Select top-k axes by predicted impact only."""
    return sorted(impacts.keys(), key=lambda x: -impacts[x])[:k]


def select_top_diversity(corr_matrix: np.ndarray, k: int) -> list[str]:
    """Select k axes by diversity only (greedy min max-correlation)."""
    n = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix)
    selected = []

    # Start with axis having lowest mean |correlation|
    mean_corr = np.mean(abs_corr, axis=1)
    first = int(np.argmin(mean_corr))
    selected.append(first)

    for _ in range(k - 1):
        best_idx = -1
        best_div = -np.inf
        for i in range(n):
            if i in selected:
                continue
            max_corr_with_sel = max(abs_corr[i, j] for j in selected)
            div = 1.0 - max_corr_with_sel
            if div > best_div:
                best_div = div
                best_idx = i
        if best_idx >= 0:
            selected.append(best_idx)

    return [AXIS_NAMES[i] for i in selected]


def select_joint(impacts: dict[str, float], corr_matrix: np.ndarray, k: int) -> list[str]:
    """
    Joint selection: impact x diversity (same logic as HypothesisSelector with
    diversity_weight=0.5). Re-implemented inline to avoid needing the full selector
    infrastructure.
    """
    abs_corr = np.abs(corr_matrix)
    n = len(AXIS_NAMES)

    # Normalize impacts
    raw = np.array([impacts.get(ax, 0.0) for ax in AXIS_NAMES])
    if raw.max() > 0:
        norm = raw / raw.max()
    else:
        norm = np.ones(n) / n

    selected = []
    w = 0.5  # diversity weight

    for _ in range(k):
        best_idx = -1
        best_score = -np.inf

        for i in range(n):
            if i in selected:
                continue

            impact = norm[i]

            if selected:
                max_corr = max(abs_corr[i, j] for j in selected)
                diversity = 1.0 - max_corr
            else:
                diversity = 1.0

            combined = (1 - w) * impact + w * diversity
            if combined > best_score:
                best_score = combined
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)

    return [AXIS_NAMES[i] for i in selected]


def measure_top_axis_hit(selected_axes: list[str], ground_truth: dict[str, float]) -> float:
    """Does the top-3 selection include the axis with highest ground-truth activity?"""
    if not ground_truth or all(v == 0 for v in ground_truth.values()):
        return 0.0
    top_axis = max(ground_truth.keys(), key=lambda x: ground_truth[x])
    return 1.0 if top_axis in selected_axes else 0.0


def measure_active_axis_coverage(
    selected_axes: list[str], ground_truth: dict[str, float]
) -> float:
    """
    Of axes with above-median ground-truth activity, what fraction do our top-k cover?
    """
    values = list(ground_truth.values())
    if not values or max(values) == 0:
        return 0.0

    median_val = np.median([v for v in values if v > 0]) if any(v > 0 for v in values) else 0.0
    active_axes = [ax for ax, v in ground_truth.items() if v > median_val]

    if not active_axes:
        return 0.0

    covered = sum(1 for ax in active_axes if ax in selected_axes)
    return covered / len(active_axes)


def measure_rank_correlation(
    predicted_impacts: dict[str, float], ground_truth: dict[str, float]
) -> float:
    """Spearman correlation between predicted and ground-truth axis rankings."""
    pred_vals = [predicted_impacts.get(ax, 0.0) for ax in AXIS_NAMES]
    true_vals = [ground_truth.get(ax, 0.0) for ax in AXIS_NAMES]

    if len(set(pred_vals)) <= 1 or len(set(true_vals)) <= 1:
        return 0.0

    rho, _ = scipy_stats.spearmanr(pred_vals, true_vals)
    return float(rho) if np.isfinite(rho) else 0.0


def measure_profile_coverage(selected_axes: list[str], ground_truth: dict[str, float]) -> float:
    """Fraction of total ground-truth activity captured by selected axes."""
    total = sum(ground_truth.values())
    if total == 0:
        return 0.0
    selected_total = sum(ground_truth.get(ax, 0.0) for ax in selected_axes)
    return selected_total / total


def phase4_evaluate(
    positions: list[dict],
    model,
    meta: dict,
    corr_matrix: np.ndarray,
) -> list[dict]:
    """Run the selector on each position and compare to ground truth."""
    p("\n" + "=" * 70)
    p("PHASE 4: Run compound selector and compare")
    p("=" * 70)

    rng = np.random.default_rng(42)
    results = []
    n_zero_context = 0
    n_pos = len(positions)

    # Pre-compute diversity-only selection (same for all positions)
    diversity_axes = select_top_diversity(corr_matrix, K)
    p(f"  Diversity-only axes: {diversity_axes}")
    p(f"  Evaluating {n_pos} positions...\n")

    for idx, pos in enumerate(positions):
        if (idx + 1) % 10 == 0 or (idx + 1) == n_pos:
            p(f"  Position {idx+1}/{n_pos}: {pos['target_id']} "
              f"({pos['n_mmps']} MMPs, {pos['n_distinct_axes']} axes)")

        ground_truth = pos["ground_truth_profile"]

        # Compute 3D context for this core (novel scaffold, not in training DB)
        ctx_vec = compute_3d_context(pos["core_smiles"])
        if np.all(ctx_vec == 0):
            n_zero_context += 1

        # Get model predictions for each axis
        impacts = probe_model_at_context(model, meta, ctx_vec)

        # Strategy 1: Joint selector (impact x diversity)
        selector_axes = select_joint(impacts, corr_matrix, K)

        # Strategy 2: Top impact only
        impact_axes = select_top_impact(impacts, K)

        # Strategy 3: Diversity only
        # (fixed, computed above)

        # Strategy 4: Random baseline (average over N_RANDOM_DRAWS)
        random_hits = []
        random_coverages = []
        random_active_coverages = []
        for _ in range(N_RANDOM_DRAWS):
            rand_idx = rng.choice(N_AXES, size=K, replace=False)
            rand_axes = [AXIS_NAMES[i] for i in rand_idx]
            random_hits.append(measure_top_axis_hit(rand_axes, ground_truth))
            random_coverages.append(measure_profile_coverage(rand_axes, ground_truth))
            random_active_coverages.append(measure_active_axis_coverage(rand_axes, ground_truth))

        results.append({
            "core_smiles": pos["core_smiles"],
            "target_id": pos["target_id"],
            "source_dataset": pos["source_dataset"],
            "n_mmps": pos["n_mmps"],
            "n_distinct_axes": pos["n_distinct_axes"],
            "zero_context": bool(np.all(ctx_vec == 0)),
            # Ground truth
            "ground_truth_profile": ground_truth,
            # Selector (joint)
            "selector_axes": selector_axes,
            "selector_top_hit": measure_top_axis_hit(selector_axes, ground_truth),
            "selector_coverage": measure_profile_coverage(selector_axes, ground_truth),
            "selector_active_coverage": measure_active_axis_coverage(selector_axes, ground_truth),
            "selector_rank_corr": measure_rank_correlation(impacts, ground_truth),
            # Impact only
            "impact_axes": impact_axes,
            "impact_top_hit": measure_top_axis_hit(impact_axes, ground_truth),
            "impact_coverage": measure_profile_coverage(impact_axes, ground_truth),
            "impact_active_coverage": measure_active_axis_coverage(impact_axes, ground_truth),
            "impact_rank_corr": measure_rank_correlation(impacts, ground_truth),
            # Diversity only
            "diversity_axes": diversity_axes,
            "diversity_top_hit": measure_top_axis_hit(diversity_axes, ground_truth),
            "diversity_coverage": measure_profile_coverage(diversity_axes, ground_truth),
            "diversity_active_coverage": measure_active_axis_coverage(diversity_axes, ground_truth),
            # Random
            "random_top_hit": float(np.mean(random_hits)),
            "random_coverage": float(np.mean(random_coverages)),
            "random_active_coverage": float(np.mean(random_active_coverages)),
            # Predicted impacts for scatter
            "predicted_impacts": impacts,
        })

    p(f"\n  Positions with zero/fallback 3D context: {n_zero_context}/{n_pos}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# Phase 5: Aggregate results and generate figure
# ══════════════════════════════════════════════════════════════════════════

def aggregate_results(results: list[dict]) -> dict:
    """Compute aggregate metrics across all positions."""
    df = pd.DataFrame(results)
    n = len(df)

    summary = {
        "n_positions": n,
        "min_mmps_threshold": MIN_MMPS_PER_POSITION,
        "min_axes_threshold": MIN_DISTINCT_AXES,
        "k": K,
        "n_random_draws": N_RANDOM_DRAWS,
        "n_zero_context": int(df["zero_context"].sum()),
    }

    # Per-dataset breakdown
    summary["per_dataset"] = {}
    for source in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == source]
        summary["per_dataset"][source] = {
            "n_positions": len(sub),
            "selector_top_hit": float(sub["selector_top_hit"].mean()),
            "selector_coverage": float(sub["selector_coverage"].mean()),
            "selector_active_coverage": float(sub["selector_active_coverage"].mean()),
            "random_top_hit": float(sub["random_top_hit"].mean()),
            "random_coverage": float(sub["random_coverage"].mean()),
        }

    # Overall strategy comparison
    strategies = {}
    for name, prefix in [
        ("joint_selector", "selector"),
        ("top_impact", "impact"),
        ("diversity_only", "diversity"),
        ("random", "random"),
    ]:
        strategies[name] = {
            "top_axis_hit_rate": float(df[f"{prefix}_top_hit"].mean()),
            "top_axis_hit_rate_se": float(df[f"{prefix}_top_hit"].std() / np.sqrt(n)) if n > 1 else 0,
            "profile_coverage_mean": float(df[f"{prefix}_coverage"].mean()),
            "profile_coverage_se": float(df[f"{prefix}_coverage"].std() / np.sqrt(n)) if n > 1 else 0,
            "active_axis_coverage_mean": float(df[f"{prefix}_active_coverage"].mean()),
            "active_axis_coverage_se": float(df[f"{prefix}_active_coverage"].std() / np.sqrt(n)) if n > 1 else 0,
        }

        if f"{prefix}_rank_corr" in df.columns:
            strategies[name]["rank_correlation_mean"] = float(df[f"{prefix}_rank_corr"].mean())
            strategies[name]["rank_correlation_std"] = float(df[f"{prefix}_rank_corr"].std())

    summary["strategies"] = strategies

    # Paired tests: selector vs each baseline
    for baseline_name, baseline_prefix in [
        ("top_impact", "impact"),
        ("diversity_only", "diversity"),
        ("random", "random"),
    ]:
        cov_diff = df["selector_coverage"] - df[f"{baseline_prefix}_coverage"]
        hit_diff = df["selector_top_hit"] - df[f"{baseline_prefix}_top_hit"]

        nonzero = cov_diff[cov_diff != 0]
        if len(nonzero) > 1:
            try:
                stat_val, p_val = scipy_stats.wilcoxon(nonzero, alternative="two-sided")
            except Exception:
                p_val = 1.0
        else:
            p_val = 1.0

        summary[f"paired_vs_{baseline_name}"] = {
            "coverage_diff_mean": float(cov_diff.mean()),
            "hit_rate_diff_mean": float(hit_diff.mean()),
            "wilcoxon_p": float(p_val),
            "selector_wins": int((cov_diff > 0).sum()),
            "ties": int((cov_diff == 0).sum()),
            "selector_loses": int((cov_diff < 0).sum()),
        }

    # Collect per-position data for scatter plot
    summary["scatter_data"] = []
    for r in results:
        for ax in AXIS_NAMES:
            summary["scatter_data"].append({
                "axis": ax,
                "predicted": r["predicted_impacts"].get(ax, 0.0),
                "ground_truth": r["ground_truth_profile"].get(ax, 0.0),
                "target_id": r["target_id"],
            })

    return summary


def make_figure(summary: dict):
    """Generate Figure 9: External validation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": False,
    })

    strategies = summary["strategies"]
    per_dataset = summary.get("per_dataset", {})

    # Panel A: Bar chart — selector vs baselines on top-axis hit rate and
    # active-axis coverage, split by dataset if enough data
    has_scatter = bool(summary.get("scatter_data"))
    if has_scatter:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    else:
        fig, ax_single = plt.subplots(figsize=(7, 5.5))
        axes = [ax_single, None]

    # --- Panel A: Strategy comparison bars ---
    strategy_labels = ["Random", "Diversity\nOnly", "Top Impact", "Joint\nSelector"]
    strategy_keys = ["random", "diversity_only", "top_impact", "joint_selector"]

    hit_rates = [strategies[k]["top_axis_hit_rate"] for k in strategy_keys]
    hit_ses = [strategies[k]["top_axis_hit_rate_se"] for k in strategy_keys]
    active_covs = [strategies[k]["active_axis_coverage_mean"] for k in strategy_keys]
    active_ses = [strategies[k]["active_axis_coverage_se"] for k in strategy_keys]

    x = np.arange(len(strategy_labels))
    width = 0.35

    blue = "#4A90D9"
    orange = "#F5A623"

    bars1 = axes[0].bar(
        x - width / 2, hit_rates, width,
        yerr=hit_ses, capsize=3,
        label="Top-axis hit rate", color=blue,
        edgecolor="white", linewidth=0.5,
    )
    bars2 = axes[0].bar(
        x + width / 2, active_covs, width,
        yerr=active_ses, capsize=3,
        label="Active-axis coverage", color=orange,
        edgecolor="white", linewidth=0.5,
    )

    axes[0].set_ylabel("Fraction", fontsize=16)
    panel_prefix = "(a) " if has_scatter else ""
    axes[0].set_title(f"{panel_prefix}Selector vs. baselines on external data", fontsize=17, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategy_labels, fontsize=13)
    axes[0].tick_params(axis='y', labelsize=12)
    y_max = max(max(hit_rates), max(active_covs)) * 1.3
    axes[0].set_ylim(0, min(y_max, 1.05))
    axes[0].legend(loc="upper left", fontsize=13, framealpha=0.9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Add value labels inside bars (avoids error bar overlap)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2, height * 0.55,
                    f"{height:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white",
                )

    # --- Panel B: Scatter of predicted rank vs ground-truth rank ---
    scatter_data = summary.get("scatter_data", [])
    if scatter_data:
        predicted = [d["predicted"] for d in scatter_data]
        ground_truth = [d["ground_truth"] for d in scatter_data]

        # Add jitter for visibility
        jitter_rng = np.random.RandomState(42)
        jx = jitter_rng.normal(0, 0.01, size=len(predicted))
        jy = jitter_rng.normal(0, 0.01, size=len(ground_truth))

        axes[1].scatter(
            np.array(predicted) + jx, np.array(ground_truth) + jy,
            alpha=0.25, s=12, c=blue, edgecolors="none",
        )

        # Compute overall Spearman
        if strategies.get("joint_selector", {}).get("rank_correlation_mean") is not None:
            rho = strategies["joint_selector"]["rank_correlation_mean"]
            rho_std = strategies["joint_selector"].get("rank_correlation_std", 0)
            axes[1].text(
                0.05, 0.95,
                f"Mean Spearman rho = {rho:.3f} +/- {rho_std:.3f}",
                transform=axes[1].transAxes, fontsize=13,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )

        # Reference line
        all_vals = predicted + ground_truth
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05
            axes[1].plot(
                [lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=1,
            )

        axes[1].set_xlabel("Predicted impact (model probe)", fontsize=16)
        axes[1].set_ylabel("Ground-truth |delta pActivity|", fontsize=16)
        axes[1].tick_params(labelsize=12)
        axes[1].set_title(
            "(b) Per-axis predicted vs. observed impact",
            fontsize=17, fontweight="bold",
        )
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

    # No suptitle — LaTeX caption handles this

    n_pos = summary["n_positions"]
    datasets = ", ".join(sorted(per_dataset.keys()))
    fig.text(
        0.5, -0.03,
        f"N = {n_pos} positions from {datasets}\n"
        f"(>= {MIN_MMPS_PER_POSITION} MMPs, >= {MIN_DISTINCT_AXES} distinct axes, novel scaffolds only)",
        ha="center", fontsize=13, style="italic", color="gray",
    )

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    p(f"\nFigure saved to {OUTPUT_FIG}")


def print_summary(summary: dict):
    """Print a human-readable summary of results."""
    p("\n" + "=" * 70)
    p("RESULTS SUMMARY")
    p("=" * 70)

    n = summary["n_positions"]
    p(f"\nN positions evaluated: {n}")
    p(f"Positions with zero 3D context: {summary['n_zero_context']}")

    p(f"\n{'Strategy':<20s}  {'Top-Hit':>10s}  {'Coverage':>10s}  {'Active Cov':>12s}  {'Rank Corr':>12s}")
    p("-" * 70)
    for name, s in summary["strategies"].items():
        rank_corr = s.get("rank_correlation_mean", "")
        if isinstance(rank_corr, float):
            rank_str = f"{rank_corr:.3f}"
        else:
            rank_str = "N/A"
        p(f"  {name:<18s}  {s['top_axis_hit_rate']:>9.3f}  "
          f"{s['profile_coverage_mean']:>9.3f}  "
          f"{s['active_axis_coverage_mean']:>11.3f}  "
          f"{rank_str:>11s}")

    # Per-dataset breakdown
    if summary.get("per_dataset"):
        p(f"\nPer-dataset breakdown:")
        for ds, d in sorted(summary["per_dataset"].items()):
            p(f"  {ds} (n={d['n_positions']}):")
            p(f"    Selector top-hit: {d['selector_top_hit']:.3f}  "
              f"coverage: {d['selector_coverage']:.3f}  "
              f"active-cov: {d['selector_active_coverage']:.3f}")
            p(f"    Random top-hit:   {d['random_top_hit']:.3f}  "
              f"coverage: {d['random_coverage']:.3f}")

    # Paired comparisons
    p(f"\nPaired comparisons (selector vs baselines):")
    for key in ["paired_vs_top_impact", "paired_vs_diversity_only", "paired_vs_random"]:
        if key in summary:
            d = summary[key]
            label = key.replace("paired_vs_", "").replace("_", " ").title()
            p(f"\n  vs {label}:")
            p(f"    Coverage diff: {d['coverage_diff_mean']:+.4f} "
              f"(W={d['selector_wins']}, T={d['ties']}, L={d['selector_loses']})")
            p(f"    Hit rate diff: {d['hit_rate_diff_mean']:+.4f}")
            p(f"    Wilcoxon p:    {d['wilcoxon_p']:.4g}")

    # Limitations
    if n < 20:
        p(f"\n*** LIMITATION: Only {n} testable positions found. "
          f"Results should be interpreted cautiously. "
          f"The external datasets may have too few compounds per target to form "
          f"enough MMPs with sufficient axis diversity. ***")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    p("=" * 70)
    p("EXTERNAL VALIDATION OF COMPOUND SELECTOR")
    p("=" * 70)
    p(f"  External data: {EXT_PARQUET}")
    p(f"  Min MMPs per position: {MIN_MMPS_PER_POSITION}")
    p(f"  Min distinct axes: {MIN_DISTINCT_AXES}")
    p(f"  K (axes to select): {K}")

    # Load external data
    p("\nLoading external validation data...")
    ext_df = pd.read_parquet(EXT_PARQUET)
    p(f"  {len(ext_df)} compounds, {ext_df['smiles'].nunique()} unique SMILES, "
      f"{ext_df['target_id'].nunique()} targets")

    # Load model and metadata
    p("\nLoading change-type model and metadata...")
    with open(CT_MODEL_PATH, "rb") as f:
        ct_model = pickle.load(f)
    with open(CT_META_PATH) as f:
        ct_meta = json.load(f)

    with open(CORR_RESULTS_PATH) as f:
        corr_data = json.load(f)
    corr_matrix = np.array(corr_data["correlation_matrix"]["matrix"])
    p(f"  Model loaded. Correlation matrix: {corr_matrix.shape}")

    sigmas = ct_meta["delta_prop_sigmas"]

    # Phase 1: Extract MMPs
    mmps_df = phase1_extract_mmps(ext_df)
    if mmps_df.empty:
        p("No MMPs extracted. Exiting.")
        return
    del ext_df

    # Phase 2: Filter novel scaffolds
    mmps_novel = phase2_filter_novel_scaffolds(mmps_df)
    del mmps_df

    if mmps_novel.empty:
        p("No novel-scaffold MMPs remain. Falling back to all MMPs...")
        # Re-run Phase 1 without scaffold filter
        ext_df = pd.read_parquet(EXT_PARQUET)
        mmps_novel = phase1_extract_mmps(ext_df)
        del ext_df
        p("*** WARNING: Using ALL MMPs (no scaffold novelty filter) ***")

    # Phase 3: Build profiles
    positions = phase3_build_profiles(mmps_novel, sigmas)
    del mmps_novel

    if not positions:
        p("\nNo positions meet criteria. Trying with relaxed thresholds...")
        # Try with all MMPs (no scaffold filter) and lower thresholds
        ext_df = pd.read_parquet(EXT_PARQUET)
        mmps_all = phase1_extract_mmps(ext_df)
        del ext_df

        # Use relaxed thresholds
        _relaxed_min_mmps = 5
        _relaxed_min_axes = 2
        p(f"  Relaxed thresholds: min_mmps={_relaxed_min_mmps}, "
          f"min_axes={_relaxed_min_axes}")
        positions = phase3_build_profiles(mmps_all, sigmas, _relaxed_min_mmps, _relaxed_min_axes)
        del mmps_all

    if not positions:
        p("\nERROR: No testable positions even with relaxed thresholds. Exiting.")
        return

    # Phase 4: Evaluate
    results = phase4_evaluate(positions, ct_model, ct_meta, corr_matrix)

    # Phase 5: Aggregate and save
    p("\n" + "=" * 70)
    p("PHASE 5: Aggregate results and generate figure")
    p("=" * 70)

    summary = aggregate_results(results)

    # Remove scatter_data from JSON (too large), save separately if needed
    scatter_for_fig = summary.pop("scatter_data", [])
    summary_for_json = summary.copy()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary_for_json, f, indent=2, default=str)
    p(f"Results saved to {OUTPUT_JSON}")

    # Put scatter_data back for figure
    summary["scatter_data"] = scatter_for_fig

    # Generate figure
    make_figure(summary)

    # Print summary
    print_summary(summary)

    elapsed = time.time() - t_start
    p(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
