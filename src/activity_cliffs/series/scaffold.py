from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class SeriesTable:
    """
    Adds a `series_id` per molecule based on Bemis–Murcko scaffold.
    """

    df: pd.DataFrame


def bemis_murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaf, canonical=True)


def assign_scaffold_series(df_curated: pd.DataFrame) -> SeriesTable:
    needed = {"target_chembl_id", "molregno", "canonical_smiles", "pActivity"}
    missing = needed - set(df_curated.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_curated.copy()
    mols = [Chem.MolFromSmiles(s) for s in df["canonical_smiles"].astype(str).tolist()]
    scaffolds = []
    for m in mols:
        scaffolds.append(bemis_murcko_scaffold_smiles(m) if m is not None else "")

    df["scaffold_smiles"] = scaffolds
    # series_id: stable integer per (target, scaffold_smiles)
    df["series_key"] = df["target_chembl_id"].astype(str) + "||" + df["scaffold_smiles"].astype(str)
    df["series_id"] = pd.factorize(df["series_key"])[0].astype(int)
    df = df.drop(columns=["series_key"])
    return SeriesTable(df=df)

