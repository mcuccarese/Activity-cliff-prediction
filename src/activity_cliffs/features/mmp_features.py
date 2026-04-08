"""
M4 Feature engineering: compute MMP features from all_mmps.parquet.

Strategy (optimised for a single-GPU workstation with ~32 GB RAM):
  RDKit is CPU-only; GPU is not used here.  The bottleneck is 25 M rows with
  376 K unique R-group SMILES and 104 K unique core SMILES.  We exploit this
  extreme repetition with a cache-first / numpy-index approach:

  Phase 1 -Load only the three SMILES columns once (~3 GB), enumerate unique
             values, then build compact numpy feature matrices:
               desc_mat  : (n_rg, 7)  float32  -7 physchem descriptors
               fp_mat    : (n_rg, 32) uint8    -256-bit Morgan FP bit-packed
               env_mat   : (n_core, 2) uint32  -attachment-env hashes r=1,2
             These matrices are tiny (< 50 MB combined).  The SMILES -> row-index
             dicts are also held in memory throughout Phase 2.

  Phase 2 -Stream 500 K-row chunks from the parquet using pyarrow iter_batches
             (reads only the 3 needed string columns per batch).  Map SMILES to
             integer indices with pandas .map(), then index into the numpy arrays
             -no per-row Python loops in the hot path.  Write each chunk
             immediately to the output parquet; never hold more than one chunk
             plus the caches in memory.

Output columns
--------------
delta_MW, delta_LogP, delta_TPSA, delta_HBDonors, delta_HBAcceptors,
delta_RotBonds, delta_HeavyAtomCount  (float32)
fp_rgroup_from, fp_rgroup_to          (fixed-size binary, 32 bytes = 256 bits)
env_hash_r1, env_hash_r2              (uint32)
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)

# Shared generator -rdFingerprintGenerator objects are thread-safe in RDKit >= 2022
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)

_ZERO_DESC: np.ndarray = np.zeros(7, dtype=np.float32)
_ZERO_FP: np.ndarray = np.zeros(32, dtype=np.uint8)

_DELTA_NAMES = [
    "delta_MW", "delta_LogP", "delta_TPSA",
    "delta_HBDonors", "delta_HBAcceptors",
    "delta_RotBonds", "delta_HeavyAtomCount",
]

_OUTPUT_SCHEMA = pa.schema([
    ("delta_MW",             pa.float32()),
    ("delta_LogP",           pa.float32()),
    ("delta_TPSA",           pa.float32()),
    ("delta_HBDonors",       pa.float32()),
    ("delta_HBAcceptors",    pa.float32()),
    ("delta_RotBonds",       pa.float32()),
    ("delta_HeavyAtomCount", pa.float32()),
    ("fp_rgroup_from",       pa.binary(32)),
    ("fp_rgroup_to",         pa.binary(32)),
    ("env_hash_r1",          pa.uint32()),
    ("env_hash_r2",          pa.uint32()),
])


# ── Per-molecule computations ─────────────────────────────────────────────────

def _rgroup_features(smi: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute descriptors and Morgan FP for one R-group SMILES in a single
    MolFromSmiles call (avoids parsing the same molecule twice).

    Returns
    -------
    desc : float32 ndarray, shape (7,)
    fp   : uint8  ndarray, shape (32,)  -256 bits packed
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return _ZERO_DESC.copy(), _ZERO_FP.copy()
    try:
        desc = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.HeavyAtomCount(mol),
            ],
            dtype=np.float32,
        )
        bv = _MORGAN_GEN.GetFingerprint(mol)
        arr = np.zeros(256, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        fp = np.packbits(arr)  # shape (32,)
        return desc, fp
    except Exception:
        return _ZERO_DESC.copy(), _ZERO_FP.copy()


def _core_env_hashes(core_smi: str) -> tuple[int, int]:
    """
    Compute atom-environment hashes (radius=1 and radius=2) around the
    attachment-point neighbour of [*:1] in the core SMILES.

    The dummy atom [*:1] marks where the R-group was cut.  Its neighbour in the
    core is the attachment atom -the atom whose local chemical environment
    determines whether a given R-group swap will be tolerated or beneficial.
    We capture this environment as a canonical SMILES fragment and MD5-hash it
    to a uint32.

    Returns (hash_r1, hash_r2).  Returns (0, 0) on any parse failure.
    """
    mol = Chem.MolFromSmiles(core_smi)
    if mol is None:
        return 0, 0

    # Locate the dummy atom [*:1] and its neighbour (the attachment atom)
    dummy_idx: Optional[int] = None
    attach_idx: Optional[int] = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_idx = atom.GetIdx()
            nbrs = list(atom.GetNeighbors())
            if nbrs:
                attach_idx = nbrs[0].GetIdx()
            break

    if attach_idx is None:
        return 0, 0

    hashes: list[int] = []
    for radius in (1, 2):
        env_bond_idxs = list(
            Chem.FindAtomEnvironmentOfRadiusN(mol, radius, attach_idx, useHs=False)
        )

        if not env_bond_idxs:
            # No bonds in environment -hash just the attachment atom symbol
            sym = mol.GetAtomWithIdx(attach_idx).GetSymbol()
            h = int(hashlib.md5(sym.encode()).hexdigest()[:8], 16) & 0xFFFF_FFFF
            hashes.append(h)
            continue

        # Collect atom and bond sets, excluding the dummy [*:1]
        atom_set: set[int] = set()
        valid_bonds: list[int] = []
        for bi in env_bond_idxs:
            b = mol.GetBondWithIdx(bi)
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if a1 != dummy_idx and a2 != dummy_idx:
                atom_set.add(a1)
                atom_set.add(a2)
                valid_bonds.append(bi)

        if not atom_set:
            sym = mol.GetAtomWithIdx(attach_idx).GetSymbol()
            h = int(hashlib.md5(sym.encode()).hexdigest()[:8], 16) & 0xFFFF_FFFF
            hashes.append(h)
            continue

        try:
            env_smi = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=list(atom_set),
                bondsToUse=valid_bonds,
                canonical=True,
            )
            h = int(hashlib.md5(env_smi.encode()).hexdigest()[:8], 16) & 0xFFFF_FFFF
        except Exception:
            h = 0
        hashes.append(h)

    return hashes[0], hashes[1]


# ── Cache builders ────────────────────────────────────────────────────────────

def build_rgroup_cache(
    unique_smiles: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Compute descriptors and Morgan FPs for all unique R-group SMILES once.

    Returns
    -------
    desc_mat   : float32 ndarray, shape (n, 7)
    fp_mat     : uint8 ndarray,   shape (n, 32)
    smi_to_idx : dict mapping each SMILES string to its row index in the matrices
    """
    n = len(unique_smiles)
    desc_mat = np.zeros((n, 7), dtype=np.float32)
    fp_mat = np.zeros((n, 32), dtype=np.uint8)
    smi_to_idx: dict[str, int] = {}

    for i, smi in enumerate(unique_smiles):
        if i % 50_000 == 0:
            logger.info("  R-group cache: %d / %d", i, n)
        smi_to_idx[smi] = i
        desc_mat[i], fp_mat[i] = _rgroup_features(smi)

    logger.info("  R-group cache: %d / %d - done", n, n)
    return desc_mat, fp_mat, smi_to_idx


def build_core_cache(
    unique_smiles: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Compute attachment-environment hashes for all unique core SMILES once.

    Returns
    -------
    env_mat    : uint32 ndarray, shape (n, 2)  -columns [hash_r1, hash_r2]
    smi_to_idx : dict mapping each SMILES string to its row index
    """
    n = len(unique_smiles)
    env_mat = np.zeros((n, 2), dtype=np.uint32)
    smi_to_idx: dict[str, int] = {}

    for i, smi in enumerate(unique_smiles):
        if i % 20_000 == 0:
            logger.info("  Core cache: %d / %d", i, n)
        smi_to_idx[smi] = i
        h1, h2 = _core_env_hashes(smi)
        env_mat[i, 0] = h1
        env_mat[i, 1] = h2

    logger.info("  Core cache: %d / %d - done", n, n)
    return env_mat, smi_to_idx


# ── Main entry point ──────────────────────────────────────────────────────────

def build_mmp_features(
    input_parquet: Path,
    output_parquet: Path,
    *,
    chunk_size: int = 500_000,
) -> None:
    """
    Read *input_parquet* (all_mmps.parquet) in streaming chunks, compute MMP
    features, and write to *output_parquet*.

    The output has the same row order as the input so rows can be aligned by
    position.  It does **not** include target_chembl_id or activity columns —
    join back on index when needed.

    Parameters
    ----------
    input_parquet  : path to all_mmps.parquet
    output_parquet : destination path (parent directory is created if needed)
    chunk_size     : rows per batch in Phase 2 (default 500 K ≈ 100–150 MB RAM)
    """
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: collect unique SMILES, build feature caches ─────────────────
    logger.info("Phase 1 - loading SMILES columns ...")
    smi_cols = pd.read_parquet(
        input_parquet,
        columns=["rgroup_from", "rgroup_to", "core_smiles"],
        engine="pyarrow",
    )
    unique_rg = pd.concat([smi_cols["rgroup_from"], smi_cols["rgroup_to"]]).unique().tolist()
    unique_core = smi_cols["core_smiles"].unique().tolist()
    total_rows = len(smi_cols)
    del smi_cols  # free ~3 GB before building caches

    logger.info(
        "  %d unique R-groups | %d unique cores | %d total rows",
        len(unique_rg), len(unique_core), total_rows,
    )

    logger.info("Phase 1 - building R-group descriptor + FP cache ...")
    desc_mat, fp_mat, rg_smi2idx = build_rgroup_cache(unique_rg)

    logger.info("Phase 1 - building core attachment-environment cache ...")
    env_mat, core_smi2idx = build_core_cache(unique_core)

    # ── Phase 2: stream chunks, index into caches, write output ──────────────
    logger.info("Phase 2 - streaming %d rows in chunks of %d ...", total_rows, chunk_size)

    pf = pq.ParquetFile(input_parquet)
    writer = pq.ParquetWriter(output_parquet, _OUTPUT_SCHEMA, compression="snappy")
    rows_done = 0

    for batch in pf.iter_batches(
        batch_size=chunk_size,
        columns=["rgroup_from", "rgroup_to", "core_smiles"],
    ):
        df = batch.to_pandas()

        # Map SMILES strings -> integer row indices in the cache matrices.
        # Unknown SMILES (shouldn't occur, but safe default) map to row 0 (zeros).
        rg_from_idx = df["rgroup_from"].map(rg_smi2idx).fillna(0).to_numpy(dtype=np.intp)
        rg_to_idx   = df["rgroup_to"].map(rg_smi2idx).fillna(0).to_numpy(dtype=np.intp)
        core_idx    = df["core_smiles"].map(core_smi2idx).fillna(0).to_numpy(dtype=np.intp)

        # Delta descriptors -pure numpy, no Python loop
        delta_desc = desc_mat[rg_to_idx] - desc_mat[rg_from_idx]  # (n, 7) float32

        # FP bytes -convert each uint8 row to a Python bytes object of length 32.
        # bytes(numpy_uint8_array) preserves all bytes including trailing nulls,
        # unlike numpy's S32 dtype which strips them.
        fp_from_rows = fp_mat[rg_from_idx]   # (n, 32) uint8
        fp_to_rows   = fp_mat[rg_to_idx]     # (n, 32) uint8
        fp_from_bytes = [bytes(fp_from_rows[i]) for i in range(len(fp_from_rows))]
        fp_to_bytes   = [bytes(fp_to_rows[i])   for i in range(len(fp_to_rows))]

        # Env hashes -numpy indexing
        env_r1 = env_mat[core_idx, 0]  # uint32
        env_r2 = env_mat[core_idx, 1]  # uint32

        table = pa.table(
            {
                "delta_MW":             pa.array(delta_desc[:, 0], type=pa.float32()),
                "delta_LogP":           pa.array(delta_desc[:, 1], type=pa.float32()),
                "delta_TPSA":           pa.array(delta_desc[:, 2], type=pa.float32()),
                "delta_HBDonors":       pa.array(delta_desc[:, 3], type=pa.float32()),
                "delta_HBAcceptors":    pa.array(delta_desc[:, 4], type=pa.float32()),
                "delta_RotBonds":       pa.array(delta_desc[:, 5], type=pa.float32()),
                "delta_HeavyAtomCount": pa.array(delta_desc[:, 6], type=pa.float32()),
                "fp_rgroup_from":       pa.array(fp_from_bytes, type=pa.binary(32)),
                "fp_rgroup_to":         pa.array(fp_to_bytes,   type=pa.binary(32)),
                "env_hash_r1":          pa.array(env_r1, type=pa.uint32()),
                "env_hash_r2":          pa.array(env_r2, type=pa.uint32()),
            },
            schema=_OUTPUT_SCHEMA,
        )
        writer.write_table(table)

        rows_done += len(df)
        logger.info(
            "  %d / %d rows (%.1f%%)", rows_done, total_rows, 100 * rows_done / total_rows
        )

    writer.close()
    logger.info("Done -> %s", output_parquet)
