"""
Prepare external validation datasets for MMP extraction.

Reads three external benchmarks (OpenFE, Schrodinger FEP, COVID Moonshot),
standardizes them into a common format, and saves a combined parquet.

Output columns:
    smiles, target_id, activity_value, activity_type, source_dataset

Usage:
    python scripts/external_validation/prepare_external_data.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit warnings for cleaner output
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────
EXT_VAL_DIR = Path("D:/Mike project data/Activity cliffs/external_validation")
OPENFE_DIR = EXT_VAL_DIR / "openfe_benchmark" / "repo" / "data"
SCHRODINGER_DIR = EXT_VAL_DIR / "schrodinger_fep" / "repo" / "fep_benchmark_inputs"
MOONSHOT_DIR = EXT_VAL_DIR / "covid_moonshot" / "repo"
OUTPUT_PATH = EXT_VAL_DIR / "combined_external.parquet"


def canonicalize_smiles(smi: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_murcko_scaffold(smi: str) -> str | None:
    """Return Murcko generic scaffold SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return None


# ── 1. OpenFE Protein-Ligand Benchmark ─────────────────────────────────
def parse_openfe() -> pd.DataFrame:
    """Parse OpenFE benchmark: YAML metadata + SDF structures per target."""
    print("=" * 60)
    print("PARSING: OpenFE Protein-Ligand Benchmark")
    print("=" * 60)

    records = []
    targets_dir = OPENFE_DIR
    target_dirs = sorted([
        d for d in targets_dir.iterdir()
        if d.is_dir()
    ])

    for target_dir in target_dirs:
        target_name = target_dir.name
        ligands_yml = target_dir / "00_data" / "ligands.yml"

        if not ligands_yml.exists():
            print(f"  SKIP {target_name}: no ligands.yml")
            continue

        with open(ligands_yml) as f:
            ligands = yaml.safe_load(f)

        if not ligands:
            print(f"  SKIP {target_name}: empty ligands.yml")
            continue

        n_valid = 0
        for lig_name, lig_data in ligands.items():
            smiles = lig_data.get("smiles")
            measurement = lig_data.get("measurement", {})
            value = measurement.get("value")
            mtype = measurement.get("type", "unknown")
            unit = measurement.get("unit", "unknown")

            if smiles is None or value is None:
                continue

            canon = canonicalize_smiles(smiles)
            if canon is None:
                continue

            # Convert to a common scale: pActivity (-log10(M))
            # Most are IC50 or Ki in uM
            if unit == "uM":
                # Convert uM to M, then -log10
                if value > 0:
                    pactivity = -np.log10(value * 1e-6)
                else:
                    continue
            elif unit == "nM":
                if value > 0:
                    pactivity = -np.log10(value * 1e-9)
                else:
                    continue
            elif unit == "kcal/mol":
                # deltaG -- store as-is, mark type
                pactivity = value
                mtype = "dG_kcal_mol"
            else:
                # Unknown unit, store raw value
                pactivity = value

            records.append({
                "smiles": canon,
                "target_id": f"openfe_{target_name}",
                "activity_value": pactivity,
                "activity_type": mtype,
                "source_dataset": "openfe_benchmark",
            })
            n_valid += 1

        print(f"  {target_name}: {n_valid} compounds")

    df = pd.DataFrame(records)
    print(f"\nOpenFE total: {len(df)} compounds across {df['target_id'].nunique()} targets")
    return df


# ── 2. Schrodinger FEP Benchmark ──────────────────────────────────────
def parse_schrodinger() -> pd.DataFrame:
    """Parse Schrodinger FEP benchmark: SDF files with r_exp_dg property."""
    print("\n" + "=" * 60)
    print("PARSING: Schrodinger FEP Benchmark")
    print("=" * 60)

    # Read metadata CSV for target info
    metadata_path = SCHRODINGER_DIR / "benchmark_metadata.csv"
    meta_df = pd.read_csv(metadata_path)

    # Build a lookup: input_file_naming_scheme -> (protein, group)
    file_to_meta = {}
    for _, row in meta_df.iterrows():
        scheme = row["Input file naming scheme"]
        protein = row["Protein"]
        group = row["Group abbreviation"]
        file_to_meta[scheme] = (protein, group)

    # Find all SDF files in structure_inputs
    sdf_base = SCHRODINGER_DIR / "structure_inputs"
    records = []

    for group_dir in sorted(sdf_base.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name

        for sdf_file in sorted(group_dir.glob("*_ligands.sdf")):
            # Extract the scheme name (remove _ligands.sdf suffix)
            scheme = sdf_file.stem.replace("_ligands", "")

            # Look up protein name from metadata
            if scheme in file_to_meta:
                protein, _ = file_to_meta[scheme]
            else:
                protein = scheme

            target_id = f"schrodinger_{group_name}_{scheme}"

            suppl = Chem.SDMolSupplier(str(sdf_file))
            n_valid = 0
            for mol in suppl:
                if mol is None:
                    continue

                smi = Chem.MolToSmiles(mol)
                if smi is None:
                    continue

                # Get experimental dG
                try:
                    dg = mol.GetDoubleProp("r_exp_dg")
                except KeyError:
                    continue

                records.append({
                    "smiles": smi,
                    "target_id": target_id,
                    "activity_value": dg,
                    "activity_type": "dG_kcal_mol",
                    "source_dataset": "schrodinger_fep",
                })
                n_valid += 1

            print(f"  {group_name}/{scheme}: {n_valid} compounds (protein={protein})")

    df = pd.DataFrame(records)
    print(f"\nSchrodinger total: {len(df)} compounds across {df['target_id'].nunique()} targets")
    return df


# ── 3. COVID Moonshot ──────────────────────────────────────────────────
def parse_covid_moonshot() -> pd.DataFrame:
    """Parse COVID Moonshot: CSV with SMILES + IC50 data."""
    print("\n" + "=" * 60)
    print("PARSING: COVID Moonshot")
    print("=" * 60)

    records = []

    for csv_name, label in [
        ("cdd_noncovalent_dates_2023_10_18_filt.csv", "noncovalent"),
        ("cdd_achiral_enantiopure_dates_2023_10_18_filt.csv", "achiral"),
    ]:
        csv_path = MOONSHOT_DIR / csv_name
        df = pd.read_csv(csv_path)

        # IC50 column name has encoding issues, just use positional
        ic50_col = df.columns[2]  # IC50_(uM)
        smiles_col = df.columns[0]  # suspected_SMILES

        print(f"\n  {csv_name}:")
        print(f"    Total rows: {len(df)}")

        n_valid = 0
        n_censored = 0
        for _, row in df.iterrows():
            smi = row[smiles_col]
            ic50_str = str(row[ic50_col]).strip()

            # Skip censored values (> or <)
            if ic50_str.startswith(">") or ic50_str.startswith("<"):
                n_censored += 1
                continue

            try:
                ic50_um = float(ic50_str)
            except (ValueError, TypeError):
                continue

            if ic50_um <= 0:
                continue

            canon = canonicalize_smiles(smi)
            if canon is None:
                continue

            # Convert IC50 (uM) to pIC50
            pic50 = -np.log10(ic50_um * 1e-6)

            records.append({
                "smiles": canon,
                "target_id": f"moonshot_mpro_{label}",
                "activity_value": pic50,
                "activity_type": "pIC50",
                "source_dataset": "covid_moonshot",
            })
            n_valid += 1

        print(f"    Valid numeric IC50: {n_valid}")
        print(f"    Censored (> or <): {n_censored}")

    df = pd.DataFrame(records)

    # Deduplicate: same SMILES may appear in both files
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["smiles", "target_id"])
    print(f"\n  Deduplicated: {before_dedup} -> {len(df)}")
    print(f"COVID Moonshot total: {len(df)} compounds")
    return df


# ── Combine and report ─────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> None:
    """Print per-target statistics."""
    print("\n" + "=" * 60)
    print("PER-TARGET STATISTICS")
    print("=" * 60)
    print(f"{'target_id':<50} {'n':>5} {'act_min':>8} {'act_max':>8} {'act_range':>9} {'scaffolds':>10}")
    print("-" * 95)

    for target_id, group in df.groupby("target_id"):
        n = len(group)
        act_min = group["activity_value"].min()
        act_max = group["activity_value"].max()
        act_range = act_max - act_min

        # Scaffold diversity
        scaffolds = set()
        for smi in group["smiles"]:
            scaf = get_murcko_scaffold(smi)
            if scaf:
                scaffolds.add(scaf)

        n_scaffolds = len(scaffolds)
        print(f"{target_id:<50} {n:>5} {act_min:>8.2f} {act_max:>8.2f} {act_range:>9.2f} {n_scaffolds:>10}")


def main():
    print("External Validation Data Preparation")
    print("=" * 60)
    print()

    # Parse all three datasets
    df_openfe = parse_openfe()
    df_schrodinger = parse_schrodinger()
    df_moonshot = parse_covid_moonshot()

    # Combine
    combined = pd.concat([df_openfe, df_schrodinger, df_moonshot], ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"COMBINED DATASET")
    print(f"{'=' * 60}")
    print(f"Total compounds: {len(combined)}")
    print(f"Unique SMILES: {combined['smiles'].nunique()}")
    print(f"Targets: {combined['target_id'].nunique()}")
    print(f"\nBy source:")
    for src, grp in combined.groupby("source_dataset"):
        print(f"  {src}: {len(grp)} compounds, {grp['target_id'].nunique()} targets")

    # Per-target stats
    compute_stats(combined)

    # Save
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
