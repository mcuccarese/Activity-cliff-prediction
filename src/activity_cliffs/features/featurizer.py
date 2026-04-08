from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class FingerprintResult:
    mols: list[Chem.Mol]
    fps: list[DataStructs.cDataStructs.ExplicitBitVect]
    valid_mask: np.ndarray  # shape (n,)


def smiles_to_mols(smiles: Iterable[str]) -> list[Optional[Chem.Mol]]:
    out: list[Optional[Chem.Mol]] = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(m)
    return out


def ecfp4_bitvect(
    mol: Chem.Mol, *, n_bits: int = 2048, use_chirality: bool = True
) -> DataStructs.cDataStructs.ExplicitBitVect:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=use_chirality)
    return gen.GetFingerprint(mol)


def featurize_ecfp4(
    smiles: Iterable[str], *, n_bits: int = 2048, use_chirality: bool = True
) -> FingerprintResult:
    mols_opt = smiles_to_mols(smiles)
    mols: list[Chem.Mol] = []
    fps: list[DataStructs.cDataStructs.ExplicitBitVect] = []
    valid = []

    for m in mols_opt:
        if m is None:
            valid.append(False)
            continue
        mols.append(m)
        fps.append(ecfp4_bitvect(m, n_bits=n_bits, use_chirality=use_chirality))
        valid.append(True)

    valid_mask = np.array(valid, dtype=bool)
    return FingerprintResult(mols=mols, fps=fps, valid_mask=valid_mask)


def rdkit_physchem_descriptors(mol: Chem.Mol) -> np.ndarray:
    # Small, stable set of common descriptors (not a huge RDKit descriptor dump).
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
        ],
        dtype=np.float32,
    )

