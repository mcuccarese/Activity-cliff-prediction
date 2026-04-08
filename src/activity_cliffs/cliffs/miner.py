from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import DataStructs

from activity_cliffs.config import CliffMiningConfig


@dataclass(frozen=True)
class CliffPairs:
    df: pd.DataFrame


def mine_activity_cliffs(
    df_series: pd.DataFrame,
    fps: list[DataStructs.cDataStructs.ExplicitBitVect],
    *,
    config: CliffMiningConfig = CliffMiningConfig(),
    series_id: Optional[int] = None,
) -> CliffPairs:
    """
    Mine activity-cliff pairs for a (target, series) slice.

    df_series must be aligned to fps (same order) and contain:
    - molregno, pActivity

    Output columns:
    - mol_i, mol_j, sim, delta_pActivity, cliff_label, cliff_score
    """
    needed = {"molregno", "pActivity"}
    missing = needed - set(df_series.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df_series) != len(fps):
        raise ValueError("df_series and fps must have the same length/order.")

    mol_ids = df_series["molregno"].to_numpy()
    p = df_series["pActivity"].astype(float).to_numpy()
    n = len(p)

    rows = []
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sims = np.asarray(sims, dtype=np.float32)

        # Candidate neighbors: above similarity threshold, exclude self
        idx = np.where((sims >= config.sim_min) & (np.arange(n) != i))[0]
        if idx.size == 0:
            continue

        # Keep only top-k neighbors by similarity (controls O(n^2))
        if idx.size > config.max_neighbors:
            topk = np.argpartition(-sims[idx], config.max_neighbors)[: config.max_neighbors]
            idx = idx[topk]

        for j in idx.tolist():
            if j <= i:
                continue  # undirected unique pairs
            delta = float(abs(p[i] - p[j]))
            sim = float(sims[j])
            cliff = int(delta >= config.delta_pactivity_min)
            score = float(delta / max(1e-6, (1.0 - sim)))
            rows.append((int(mol_ids[i]), int(mol_ids[j]), sim, delta, cliff, score))

    out = pd.DataFrame(
        rows,
        columns=["mol_i", "mol_j", "sim", "delta_pActivity", "cliff_label", "cliff_score"],
    )

    if series_id is not None:
        out.insert(0, "series_id", int(series_id))

    return CliffPairs(df=out)

