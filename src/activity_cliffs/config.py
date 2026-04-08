from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CliffMiningConfig:
    # Similarity threshold for candidate pairs (ECFP4/Tanimoto)
    sim_min: float = 0.85
    # Minimum activity jump in log units (pIC50 / pKi / pEC50)
    delta_pactivity_min: float = 1.5
    # Maximum neighbors to consider per molecule (keeps pair explosion manageable)
    max_neighbors: int = 200


@dataclass(frozen=True)
class ChemblConfig:
    sqlite_path: Path
    # Standard bioactivity types to consider for mining.
    allowed_standard_types: tuple[str, ...] = ("IC50", "Ki", "EC50")
    # Require target confidence score >= this (ChEMBL target assignment confidence).
    min_confidence_score: int = 7

