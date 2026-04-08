"""
M6b: Classify R-group fragments into medchem-meaningful change types.

For each R-group SMILES (with [*:1] attachment marker), compute a property
vector describing its chemical character.  For each MMP transform
(rgroup_from -> rgroup_to), the change-type vector is simply:

    change_type = props(rgroup_to) - props(rgroup_from)

This gives within-group variance: all transforms from the same molecule share
the same 3D context, but different R-group swaps produce different change-type
vectors.  The context x change_type cross-product is the key interaction feature.

Property vector per R-group (11 dimensions):
  0. has_ewg       - contains electron-withdrawing group (F, Cl, Br, CF3, NO2, CN, SO2, COOH)
  1. has_edg       - contains electron-donating group (NH2, OH, OMe, NMe2, alkyl donors)
  2. ewg_count     - number of EWG motifs matched
  3. edg_count     - number of EDG motifs matched
  4. n_hbd         - H-bond donor count
  5. n_hba         - H-bond acceptor count
  6. lipophilicity - MolLogP (continuous lipophilic character)
  7. heavy_atoms   - heavy atom count (size proxy)
  8. n_rings       - total ring count
  9. n_arom_rings  - aromatic ring count
 10. fsp3          - fraction of sp3 carbons (3D character / flatness)

Change-type vector (11 dimensions, same order):
  delta_has_ewg, delta_has_edg, delta_ewg_count, delta_edg_count,
  delta_n_hbd, delta_n_hba, delta_lipophilicity, delta_heavy_atoms,
  delta_n_rings, delta_n_arom_rings, delta_fsp3
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# ── SMARTS definitions for EWG / EDG classification ──────────────────────────

# Electron-withdrawing groups
_EWG_SMARTS = [
    "[F]",                    # fluorine
    "[Cl]",                   # chlorine
    "[Br]",                   # bromine
    "[$(C(F)(F)F)]",          # CF3
    "[$(N(=O)~O)]",           # nitro
    "[C]#[N]",                # nitrile
    "[$(S(=O)(=O))]",         # sulfonyl / sulfone
    "[$(C(=O)O)]",            # carboxylic acid / ester
    "[$(C(=O)F),$(C(=O)Cl)]", # acyl halide
    "[$(C=O)]",               # carbonyl (general, weak EWG)
]

# Electron-donating groups
_EDG_SMARTS = [
    "[NH2]",                  # primary amine
    "[OH]",                   # hydroxyl
    "[$(OC)]",                # ether / methoxy (O bonded to C)
    "[$(N(C)C)]",             # tertiary amine (NR2)
    "[$(NC)]",                # secondary amine (NHR)
]

# Pre-compile SMARTS patterns
_EWG_PATTERNS: list[Optional[Chem.Mol]] = []
_EDG_PATTERNS: list[Optional[Chem.Mol]] = []


def _init_patterns() -> None:
    """Compile SMARTS patterns once on first use."""
    global _EWG_PATTERNS, _EDG_PATTERNS
    if _EWG_PATTERNS:
        return
    for sma in _EWG_SMARTS:
        pat = Chem.MolFromSmarts(sma)
        if pat is not None:
            _EWG_PATTERNS.append(pat)
        else:
            logger.warning("Failed to compile EWG SMARTS: %s", sma)
    for sma in _EDG_SMARTS:
        pat = Chem.MolFromSmarts(sma)
        if pat is not None:
            _EDG_PATTERNS.append(pat)
        else:
            logger.warning("Failed to compile EDG SMARTS: %s", sma)


# ── Feature names ────────────────────────────────────────────────────────────

RGROUP_PROP_NAMES = [
    "has_ewg",
    "has_edg",
    "ewg_count",
    "edg_count",
    "n_hbd",
    "n_hba",
    "lipophilicity",
    "heavy_atoms",
    "n_rings",
    "n_arom_rings",
    "fsp3",
]

CHANGE_TYPE_NAMES = [f"delta_{name}" for name in RGROUP_PROP_NAMES]

N_PROPS = len(RGROUP_PROP_NAMES)
_ZERO_PROPS = np.zeros(N_PROPS, dtype=np.float32)


# ── Per-R-group computation ──────────────────────────────────────────────────

def _cap_rgroup(smi: str) -> Optional[Chem.Mol]:
    """Parse R-group SMILES, replace [*:1] dummy with H, return mol."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # Replace dummy atom ([*:1], atomic num 0) with hydrogen
    ed = Chem.RWMol(mol)
    for atom in ed.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)
            break

    try:
        capped = ed.GetMol()
        Chem.SanitizeMol(capped)
        return capped
    except Exception:
        return None


def compute_rgroup_props(smi: str) -> np.ndarray:
    """
    Compute the 11-dim property vector for one R-group SMILES.

    Parameters
    ----------
    smi : str
        R-group SMILES with [*:1] attachment marker.

    Returns
    -------
    props : float32 ndarray, shape (11,)
    """
    _init_patterns()

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return _ZERO_PROPS.copy()

    # Use original mol (with dummy) for substructure matching — the dummy
    # doesn't interfere with SMARTS matching of functional groups
    ewg_count = 0
    for pat in _EWG_PATTERNS:
        ewg_count += len(mol.GetSubstructMatches(pat))
    has_ewg = float(ewg_count > 0)

    edg_count = 0
    for pat in _EDG_PATTERNS:
        edg_count += len(mol.GetSubstructMatches(pat))
    has_edg = float(edg_count > 0)

    # Use capped mol (dummy → H) for property calculations
    capped = _cap_rgroup(smi)
    if capped is None:
        # Fall back: just return EWG/EDG flags from the original mol
        props = _ZERO_PROPS.copy()
        props[0] = has_ewg
        props[1] = has_edg
        props[2] = float(ewg_count)
        props[3] = float(edg_count)
        return props

    try:
        n_hbd = float(Descriptors.NumHDonors(capped))
        n_hba = float(Descriptors.NumHAcceptors(capped))
        logp = float(Descriptors.MolLogP(capped))
        hac = float(Descriptors.HeavyAtomCount(capped))
        n_rings = float(rdMolDescriptors.CalcNumRings(capped))
        n_arom = float(rdMolDescriptors.CalcNumAromaticRings(capped))
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(capped))
    except Exception:
        return _ZERO_PROPS.copy()

    return np.array(
        [
            has_ewg,
            has_edg,
            float(ewg_count),
            float(edg_count),
            n_hbd,
            n_hba,
            logp,
            hac,
            n_rings,
            n_arom,
            fsp3,
        ],
        dtype=np.float32,
    )


# ── Cache builder (same pattern as M4 / M6a) ────────────────────────────────

def build_rgroup_prop_cache(
    unique_rgroups: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Compute property vectors for all unique R-group SMILES.

    Returns
    -------
    prop_mat   : float32 ndarray, shape (n, 11)
    smi_to_idx : dict mapping R-group SMILES -> row index
    """
    n = len(unique_rgroups)
    prop_mat = np.zeros((n, N_PROPS), dtype=np.float32)
    smi_to_idx: dict[str, int] = {}
    n_failed = 0

    for i, smi in enumerate(unique_rgroups):
        if i % 50_000 == 0:
            logger.info("  R-group props: %d / %d (%.1f%%)", i, n, 100 * i / n)
        smi_to_idx[smi] = i
        props = compute_rgroup_props(smi)
        prop_mat[i] = props
        if np.all(props == 0):
            n_failed += 1

    logger.info(
        "  R-group props: %d / %d done (%d all-zero)", n, n, n_failed
    )
    return prop_mat, smi_to_idx
