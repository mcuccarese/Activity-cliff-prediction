"""
Check scaffold overlap between external validation data and ChEMBL training set.

Extracts Murcko scaffolds from both the external validation compounds and our
ChEMBL training data (from MMP parquet files), then computes overlap statistics.

This answers: "How many external scaffolds are truly novel vs. seen in training?"

Usage:
    python scripts/external_validation/check_scaffold_overlap.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXT_VAL_DIR = PROJECT_ROOT / "outputs" / "external_validation"
COMBINED_PATH = EXT_VAL_DIR / "combined_external.parquet"
MMP_DIR = PROJECT_ROOT / "outputs" / "mmps"
OUTPUT_PATH = EXT_VAL_DIR / "scaffold_overlap_report.csv"


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


def load_training_smiles() -> set[str]:
    """Load all unique SMILES from ChEMBL training MMP files."""
    print("Loading training SMILES from MMP parquet files...")
    all_smiles = set()

    # Read each target's MMP file and collect unique SMILES
    target_dirs = sorted([
        d for d in MMP_DIR.iterdir()
        if d.is_dir() and (d / "mmps.parquet").exists()
    ])

    for target_dir in target_dirs:
        parquet_path = target_dir / "mmps.parquet"
        try:
            # Only read SMILES columns to save memory
            df = pd.read_parquet(parquet_path, columns=["smiles_from", "smiles_to"])
            all_smiles.update(df["smiles_from"].dropna().unique())
            all_smiles.update(df["smiles_to"].dropna().unique())
            print(f"  {target_dir.name}: +{len(df['smiles_from'].unique()) + len(df['smiles_to'].unique())} SMILES entries")
        except Exception as e:
            print(f"  {target_dir.name}: ERROR - {e}")

    print(f"\nTotal unique training SMILES: {len(all_smiles)}")
    return all_smiles


def extract_scaffolds(smiles_set: set[str] | list[str], label: str) -> set[str]:
    """Extract Murcko generic scaffolds from a set of SMILES."""
    print(f"\nExtracting scaffolds from {label}...")
    scaffolds = set()
    n_failed = 0
    total = len(smiles_set)

    for i, smi in enumerate(smiles_set):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{total}...")
        scaf = get_murcko_scaffold(smi)
        if scaf:
            scaffolds.add(scaf)
        else:
            n_failed += 1

    print(f"  {label}: {len(scaffolds)} unique scaffolds from {total} SMILES ({n_failed} failed)")
    return scaffolds


def main():
    print("=" * 60)
    print("SCAFFOLD OVERLAP ANALYSIS")
    print("External validation vs. ChEMBL training set")
    print("=" * 60)

    # ── Load external validation data ──
    print("\nLoading external validation data...")
    ext_df = pd.read_parquet(COMBINED_PATH)
    print(f"  {len(ext_df)} compounds, {ext_df['smiles'].nunique()} unique SMILES")

    # ── Load training data ──
    training_smiles = load_training_smiles()

    # ── Extract scaffolds ──
    ext_scaffolds = extract_scaffolds(ext_df["smiles"].unique(), "external validation")
    training_scaffolds = extract_scaffolds(training_smiles, "ChEMBL training")

    # ── Compute overlap ──
    overlapping = ext_scaffolds & training_scaffolds
    novel = ext_scaffolds - training_scaffolds

    print("\n" + "=" * 60)
    print("OVERALL SCAFFOLD OVERLAP")
    print("=" * 60)
    print(f"  External scaffolds:     {len(ext_scaffolds)}")
    print(f"  Training scaffolds:     {len(training_scaffolds)}")
    print(f"  Overlapping:            {len(overlapping)}")
    print(f"  Novel (external only):  {len(novel)}")
    print(f"  Percent novel:          {100 * len(novel) / len(ext_scaffolds):.1f}%")

    # ── Per-dataset breakdown ──
    print("\n" + "=" * 60)
    print("PER-DATASET BREAKDOWN")
    print("=" * 60)

    report_rows = []

    for source in sorted(ext_df["source_dataset"].unique()):
        source_df = ext_df[ext_df["source_dataset"] == source]
        source_smiles = source_df["smiles"].unique()
        source_scaffolds = set()
        for smi in source_smiles:
            scaf = get_murcko_scaffold(smi)
            if scaf:
                source_scaffolds.add(scaf)

        src_overlap = source_scaffolds & training_scaffolds
        src_novel = source_scaffolds - training_scaffolds
        pct_novel = 100 * len(src_novel) / len(source_scaffolds) if source_scaffolds else 0

        print(f"\n  {source}:")
        print(f"    Compounds: {len(source_smiles)}")
        print(f"    Scaffolds: {len(source_scaffolds)}")
        print(f"    Overlapping: {len(src_overlap)}")
        print(f"    Novel: {len(src_novel)} ({pct_novel:.1f}%)")

        report_rows.append({
            "source_dataset": source,
            "n_compounds": len(source_smiles),
            "n_scaffolds": len(source_scaffolds),
            "n_overlapping": len(src_overlap),
            "n_novel": len(src_novel),
            "percent_novel": round(pct_novel, 1),
        })

    # ── Per-target breakdown ──
    print("\n" + "=" * 60)
    print("PER-TARGET BREAKDOWN")
    print("=" * 60)
    print(f"{'target_id':<55} {'n_cmpd':>6} {'n_scaf':>6} {'n_ovlp':>6} {'n_novel':>7} {'%novel':>7}")
    print("-" * 90)

    for target_id in sorted(ext_df["target_id"].unique()):
        target_df = ext_df[ext_df["target_id"] == target_id]
        target_smiles = target_df["smiles"].unique()
        target_scaffolds = set()
        for smi in target_smiles:
            scaf = get_murcko_scaffold(smi)
            if scaf:
                target_scaffolds.add(scaf)

        if not target_scaffolds:
            continue

        tgt_overlap = target_scaffolds & training_scaffolds
        tgt_novel = target_scaffolds - training_scaffolds
        pct_novel = 100 * len(tgt_novel) / len(target_scaffolds)

        print(f"{target_id:<55} {len(target_smiles):>6} {len(target_scaffolds):>6} "
              f"{len(tgt_overlap):>6} {len(tgt_novel):>7} {pct_novel:>6.1f}%")

        report_rows.append({
            "source_dataset": target_id,
            "n_compounds": len(target_smiles),
            "n_scaffolds": len(target_scaffolds),
            "n_overlapping": len(tgt_overlap),
            "n_novel": len(tgt_novel),
            "percent_novel": round(pct_novel, 1),
        })

    # Add overall summary row
    report_rows.insert(0, {
        "source_dataset": "OVERALL",
        "n_compounds": ext_df["smiles"].nunique(),
        "n_scaffolds": len(ext_scaffolds),
        "n_overlapping": len(overlapping),
        "n_novel": len(novel),
        "percent_novel": round(100 * len(novel) / len(ext_scaffolds), 1),
    })

    # Save report
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nReport saved to: {OUTPUT_PATH}")

    # ── Compound-level overlap (exact SMILES match) ──
    print("\n" + "=" * 60)
    print("EXACT COMPOUND OVERLAP (same SMILES in training)")
    print("=" * 60)
    ext_unique = set(ext_df["smiles"].unique())
    exact_overlap = ext_unique & training_smiles
    exact_novel = ext_unique - training_smiles
    print(f"  External unique SMILES:  {len(ext_unique)}")
    print(f"  Exact matches in training: {len(exact_overlap)}")
    print(f"  Novel compounds:           {len(exact_novel)}")
    print(f"  Percent novel compounds:   {100 * len(exact_novel) / len(ext_unique):.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
