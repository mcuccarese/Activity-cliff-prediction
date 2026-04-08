from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _ecfp4_numpy(smiles: str, *, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=True)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class PairDataset(Dataset):
    def __init__(self, x_i: np.ndarray, x_j: np.ndarray, y_cliff: np.ndarray):
        self.x_i = torch.from_numpy(x_i.astype(np.float32, copy=False))
        self.x_j = torch.from_numpy(x_j.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y_cliff.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x_i[idx], self.x_j[idx], self.y[idx]


class MLPEncoder(nn.Module):
    def __init__(self, n_bits: int = 2048, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


def contrastive_loss(dist: torch.Tensor, y_cliff: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Classic contrastive loss:
    - non-cliff (0): minimize distance
    - cliff (1): push distance above margin
    """
    y = y_cliff
    loss_noncliff = (1.0 - y) * torch.pow(dist, 2)
    loss_cliff = y * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return (loss_noncliff + loss_cliff).mean()


def build_pair_molecule_arrays(
    df_series: pd.DataFrame, df_pairs: pd.DataFrame, *, n_bits: int = 2048
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smiles_by_mol = dict(zip(df_series["molregno"].astype(int), df_series["canonical_smiles"].astype(str)))
    cache: dict[int, np.ndarray] = {}

    def fp(molregno: int) -> np.ndarray | None:
        if molregno in cache:
            return cache[molregno]
        smi = smiles_by_mol.get(int(molregno))
        if not smi:
            return None
        arr = _ecfp4_numpy(smi, n_bits=n_bits)
        if arr is None:
            return None
        cache[int(molregno)] = arr
        return arr

    x_i, x_j, y, groups = [], [], [], []
    for r in df_pairs.itertuples(index=False):
        a = fp(int(r.mol_i))
        b = fp(int(r.mol_j))
        if a is None or b is None:
            continue
        x_i.append(a)
        x_j.append(b)
        y.append(int(r.cliff_label))
        groups.append(int(r.series_id))

    return (
        np.stack(x_i, axis=0),
        np.stack(x_j, axis=0),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


@dataclass(frozen=True)
class ContrastiveArtifacts:
    model: MLPEncoder
    metrics: Dict[str, float]


def train_contrastive_encoder(
    df_series: pd.DataFrame,
    df_pairs: pd.DataFrame,
    *,
    emb_dim: int = 128,
    margin: float = 1.0,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs: int = 8,
    random_state: int = 0,
) -> ContrastiveArtifacts:
    x_i, x_j, y, groups = build_pair_molecule_arrays(df_series, df_pairs)
    if len(y) < 500:
        raise ValueError(f"Not enough pairs for contrastive training ({len(y)}).")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(x_i, y, groups=groups))

    ds_train = PairDataset(x_i[train_idx], x_j[train_idx], y[train_idx])
    ds_test = PairDataset(x_i[test_idx], x_j[test_idx], y[test_idx])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPEncoder(n_bits=x_i.shape[1], emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for a, b, yb in dl_train:
            a = a.to(device)
            b = b.to(device)
            yb = yb.to(device)
            za = model(a)
            zb = model(b)
            dist = torch.linalg.norm(za - zb, dim=-1)
            loss = contrastive_loss(dist, yb, margin=margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Evaluate: distance should separate cliffs from non-cliffs.
    model.eval()
    all_y, all_score = [], []
    with torch.no_grad():
        for a, b, yb in dl_test:
            a = a.to(device)
            b = b.to(device)
            za = model(a)
            zb = model(b)
            dist = torch.linalg.norm(za - zb, dim=-1).cpu().numpy()
            # larger distance => more cliff-like
            all_score.append(dist)
            all_y.append(yb.numpy())

    y_test = np.concatenate(all_y, axis=0)
    score = np.concatenate(all_score, axis=0)

    metrics: Dict[str, float] = {
        "n_pairs_total": float(len(y)),
        "n_pairs_test": float(len(y_test)),
        "cliff_rate_total": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "pr_auc": float(average_precision_score(y_test, score)),
    }
    return ContrastiveArtifacts(model=model, metrics=metrics)

