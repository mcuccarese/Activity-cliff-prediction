from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def _ecfp4_bitvect(smiles: str, *, n_bits: int = 2048) -> DataStructs.cDataStructs.ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=True)
    return gen.GetFingerprint(mol)


def bitvect_to_numpy(fp: DataStructs.cDataStructs.ExplicitBitVect) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def pair_feature_xor(fp_i: np.ndarray, fp_j: np.ndarray) -> np.ndarray:
    # Captures which bits changed between i and j.
    return np.bitwise_xor(fp_i, fp_j).astype(np.int8, copy=False)


@dataclass(frozen=True)
class BaselineArtifacts:
    clf: LogisticRegression
    reg: Ridge
    metrics: Dict[str, float]


def build_pair_dataset(
    df_series: pd.DataFrame, df_pairs: pd.DataFrame, *, n_bits: int = 2048
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds features for (mol_i, mol_j) pairs.

    Returns:
    - X: (n_pairs, n_bits) XOR fingerprint
    - y_cliff: (n_pairs,)
    - y_delta: (n_pairs,)
    - groups: (n_pairs,) series_id for leakage-safe split
    """
    smiles_by_mol = dict(zip(df_series["molregno"].astype(int), df_series["canonical_smiles"].astype(str)))

    fp_cache: dict[int, np.ndarray] = {}
    def fp_for(molregno: int) -> np.ndarray | None:
        if molregno in fp_cache:
            return fp_cache[molregno]
        smi = smiles_by_mol.get(int(molregno))
        if not smi:
            return None
        fp = _ecfp4_bitvect(smi, n_bits=n_bits)
        if fp is None:
            return None
        arr = bitvect_to_numpy(fp)
        fp_cache[int(molregno)] = arr
        return arr

    feats = []
    y_cliff = []
    y_delta = []
    groups = []

    for r in df_pairs.itertuples(index=False):
        fp_i = fp_for(int(r.mol_i))
        fp_j = fp_for(int(r.mol_j))
        if fp_i is None or fp_j is None:
            continue
        feats.append(pair_feature_xor(fp_i, fp_j))
        y_cliff.append(int(r.cliff_label))
        y_delta.append(float(r.delta_pActivity))
        groups.append(int(r.series_id))

    X = np.stack(feats, axis=0) if feats else np.zeros((0, n_bits), dtype=np.int8)
    return X, np.asarray(y_cliff, dtype=np.int64), np.asarray(y_delta, dtype=np.float32), np.asarray(groups, dtype=np.int64)


def train_baselines_pairwise(
    df_series: pd.DataFrame,
    df_pairs: pd.DataFrame,
    *,
    random_state: int = 0,
) -> BaselineArtifacts:
    X, y_cliff, y_delta, groups = build_pair_dataset(df_series, df_pairs)
    if len(y_cliff) < 200:
        raise ValueError(f"Not enough pairs after featurization ({len(y_cliff)}). Try a bigger target/looser thresholds.")

    # Group split by series_id (prevents scaffold/series leakage).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y_cliff, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_cliff[train_idx], y_cliff[test_idx]
    d_train, d_test = y_delta[train_idx], y_delta[test_idx]

    clf = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
        solver="saga",
    )
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]

    # Regress delta with ridge on same features (predict magnitude, not label).
    reg = Ridge(alpha=1.0, random_state=random_state)
    reg.fit(X_train.astype(np.float32), d_train)
    d_pred = reg.predict(X_test.astype(np.float32))

    metrics: Dict[str, float] = {
        "n_pairs_total": float(len(y_cliff)),
        "n_pairs_test": float(len(y_test)),
        "cliff_rate_total": float(y_cliff.mean()),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "delta_mae": float(mean_absolute_error(d_test, d_pred)),
    }
    return BaselineArtifacts(clf=clf, reg=reg, metrics=metrics)

