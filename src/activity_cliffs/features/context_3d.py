"""
M6a: 3D pharmacophore context features at MMP attachment points.

For each unique core SMILES (scaffold with [*:1] attachment marker), generate
a 3D conformer and compute interpretable local features at the attachment atom:

  0. n_donor_4A       - H-bond donors within 4Å of attachment
  1. n_acceptor_4A    - H-bond acceptors within 4Å
  2. n_hydrophobic_4A - Hydrophobic atoms within 4Å
  3. n_aromatic_4A    - Aromatic atoms within 4Å
  4. sasa_attach      - Solvent-accessible surface area at attachment atom (Å²)
  5. gasteiger_charge  - Gasteiger partial charge at attachment atom
  6. n_rotbonds_2     - Rotatable bonds within 2 bonds of attachment
  7. is_aromatic_attach - Whether attachment atom is in an aromatic ring (0/1)
  8. n_heavy_4A       - Total heavy atoms within 4Å (steric crowding)

Strategy: compute once per unique core_smiles (~104K), then look up by index
when joining back to the 25M-row MMP table (same cache-first pattern as M4).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA

logger = logging.getLogger(__name__)

CONTEXT_3D_FEATURES = [
    "n_donor_4A",
    "n_acceptor_4A",
    "n_hydrophobic_4A",
    "n_aromatic_4A",
    "sasa_attach",
    "gasteiger_charge",
    "n_rotbonds_2",
    "is_aromatic_attach",
    "n_heavy_4A",
]

N_FEATURES = len(CONTEXT_3D_FEATURES)
_ZERO_FEATURES = np.zeros(N_FEATURES, dtype=np.float32)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_attachment(mol: Chem.Mol) -> tuple[Optional[int], Optional[int]]:
    """Return (dummy_idx, attach_idx) for [*:1] in a core molecule."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_idx = atom.GetIdx()
            nbrs = list(atom.GetNeighbors())
            if nbrs:
                return dummy_idx, nbrs[0].GetIdx()
            return dummy_idx, None
    return None, None


def _prepare_for_embedding(
    mol: Chem.Mol, dummy_idx: int
) -> Chem.Mol:
    """Replace [*:1] dummy with H and add explicit Hs for 3D embedding."""
    ed = Chem.RWMol(mol)
    dat = ed.GetAtomWithIdx(dummy_idx)
    dat.SetAtomicNum(1)
    dat.SetAtomMapNum(0)
    dat.SetIsotope(0)
    dat.SetNoImplicit(False)
    mol2 = ed.GetMol()
    return Chem.AddHs(mol2)


def _classify_pharmacophore(atom: Chem.Atom) -> str:
    """Classify a heavy atom into donor / acceptor / hydrophobic / aromatic."""
    anum = atom.GetAtomicNum()
    if anum <= 1:
        return "skip"

    if atom.GetIsAromatic():
        return "aromatic"

    # N, O with H → donor (also acceptor, but donor dominates for medchem)
    # After AddHs(), GetTotalNumHs() may be 0 — count bonded H atoms directly
    if anum in (7, 8):
        n_h = atom.GetTotalNumHs() + sum(
            1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1
        )
        return "donor" if n_h > 0 else "acceptor"

    # S, P → acceptor (lone pairs)
    if anum in (15, 16):
        return "acceptor"

    # Halogens → weak acceptor
    if anum in (9, 17, 35, 53):
        return "acceptor"

    # Carbon → hydrophobic if not bonded to a heteroatom
    if anum == 6:
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() not in (1, 6):
                return "polar_carbon"
        return "hydrophobic"

    return "other"


def _rotatable_bonds_near(
    mol: Chem.Mol, center_idx: int, max_bond_dist: int = 2
) -> int:
    """Count rotatable bonds within *max_bond_dist* bonds of *center_idx*."""
    rot_smarts = Chem.MolFromSmarts(
        "[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]"
    )
    matches = mol.GetSubstructMatches(rot_smarts)
    if not matches:
        return 0

    count = 0
    for a1, a2 in matches:
        d1 = 0 if a1 == center_idx else len(Chem.GetShortestPath(mol, center_idx, a1)) - 1
        d2 = 0 if a2 == center_idx else len(Chem.GetShortestPath(mol, center_idx, a2)) - 1
        if min(d1, d2) <= max_bond_dist:
            count += 1
    return count


# ── Per-core computation ─────────────────────────────────────────────────────

def compute_3d_context(core_smi: str) -> np.ndarray:
    """
    Compute 3D pharmacophore context features for one core SMILES.

    Parameters
    ----------
    core_smi : str
        Core SMILES with [*:1] attachment marker.

    Returns
    -------
    features : float32 ndarray, shape (9,)
        See CONTEXT_3D_FEATURES for column names.
    """
    mol = Chem.MolFromSmiles(core_smi)
    if mol is None:
        return _ZERO_FEATURES.copy()

    dummy_idx, attach_idx = _find_attachment(mol)
    if attach_idx is None:
        return _ZERO_FEATURES.copy()

    # ── Topological features (no 3D needed) ──────────────────────────────────
    is_aromatic = float(mol.GetAtomWithIdx(attach_idx).GetIsAromatic())
    n_rotbonds = float(_rotatable_bonds_near(mol, attach_idx, max_bond_dist=2))

    # ── Generate capped molecule (dummy → H) for charge + 3D ──────────────
    mol3d = _prepare_for_embedding(mol, dummy_idx)

    # ── Gasteiger charge (on capped mol — dummy atom causes NaN) ─────────
    try:
        # Work on a 2D copy without explicit Hs for cleaner charge calc
        mol_capped = Chem.RWMol(mol)
        mol_capped.GetAtomWithIdx(dummy_idx).SetAtomicNum(1)
        mol_capped.GetAtomWithIdx(dummy_idx).SetAtomMapNum(0)
        mol_capped.GetAtomWithIdx(dummy_idx).SetIsotope(0)
        mol_capped = mol_capped.GetMol()
        AllChem.ComputeGasteigerCharges(mol_capped)
        charge = float(
            mol_capped.GetAtomWithIdx(attach_idx).GetDoubleProp("_GasteigerCharge")
        )
        if not np.isfinite(charge):
            charge = 0.0
    except Exception:
        charge = 0.0

    # attach_idx is still valid: we only changed the dummy atom in-place and
    # appended explicit Hs at the end — heavy-atom indices are preserved.

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 200

    embed_ok = AllChem.EmbedMolecule(mol3d, params)
    if embed_ok != 0:
        # Fallback: distance-geometry without ETKDG refinement
        embed_ok = AllChem.EmbedMolecule(mol3d, randomSeed=42)

    if embed_ok != 0:
        # Cannot embed — return topological-only features
        feat = _ZERO_FEATURES.copy()
        feat[5] = charge
        feat[6] = n_rotbonds
        feat[7] = is_aromatic
        return feat

    # MMFF geometry optimisation
    try:
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=200)
    except Exception:
        pass  # use unoptimised conformer

    conf = mol3d.GetConformer()
    attach_pos = np.array(conf.GetAtomPosition(attach_idx))

    # ── Pharmacophore environment within 4 Å ─────────────────────────────────
    n_donor = 0
    n_acceptor = 0
    n_hydrophobic = 0
    n_aromatic_4a = 0
    n_heavy = 0

    for atom in mol3d.GetAtoms():
        idx = atom.GetIdx()
        if idx == attach_idx or atom.GetAtomicNum() <= 1:
            continue

        pos = np.array(conf.GetAtomPosition(idx))
        dist = np.linalg.norm(pos - attach_pos)
        if dist > 4.0:
            continue

        n_heavy += 1
        ptype = _classify_pharmacophore(atom)
        if ptype == "donor":
            n_donor += 1
        elif ptype == "acceptor":
            n_acceptor += 1
        elif ptype == "hydrophobic":
            n_hydrophobic += 1
        elif ptype == "aromatic":
            n_aromatic_4a += 1

    # ── SASA at attachment atom ───────────────────────────────────────────────
    try:
        radii = rdFreeSASA.classifyAtoms(mol3d)
        rdFreeSASA.CalcSASA(mol3d, radii)
        sasa = float(
            mol3d.GetAtomWithIdx(attach_idx).GetDoubleProp("SASA")
        )
    except Exception:
        sasa = 0.0

    return np.array(
        [
            float(n_donor),
            float(n_acceptor),
            float(n_hydrophobic),
            float(n_aromatic_4a),
            sasa,
            charge,
            n_rotbonds,
            is_aromatic,
            float(n_heavy),
        ],
        dtype=np.float32,
    )


# ── Cache builder (same pattern as M4) ───────────────────────────────────────

def build_context_3d_cache(
    unique_cores: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Compute 3D context features for every unique core SMILES.

    Returns
    -------
    context_mat : float32 ndarray, shape (n, 9)
    smi_to_idx  : dict mapping core SMILES → row index in context_mat
    """
    n = len(unique_cores)
    context_mat = np.zeros((n, N_FEATURES), dtype=np.float32)
    smi_to_idx: dict[str, int] = {}
    n_failed = 0

    for i, smi in enumerate(unique_cores):
        if i % 2_000 == 0:
            logger.info("  3D context: %d / %d (%.1f%%)", i, n, 100 * i / n)
        smi_to_idx[smi] = i
        feat = compute_3d_context(smi)
        context_mat[i] = feat
        if np.all(feat == 0):
            n_failed += 1

    logger.info(
        "  3D context: %d / %d done (%d failed embedding)",
        n,
        n,
        n_failed,
    )
    return context_mat, smi_to_idx
