from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS


@dataclass(frozen=True)
class SeriesVizInputs:
    df_series: pd.DataFrame  # must include: molregno, pActivity
    df_pairs: pd.DataFrame  # must include: mol_i, mol_j, cliff_label


def plot_cliff_network(
    df_series: pd.DataFrame,
    df_pairs: pd.DataFrame,
    *,
    title: str,
    outpath: Path,
    only_cliffs: bool = True,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df_edges = df_pairs.copy()
    if only_cliffs:
        df_edges = df_edges[df_edges["cliff_label"] == 1]

    G = nx.Graph()
    for r in df_series.itertuples(index=False):
        G.add_node(int(r.molregno), pActivity=float(r.pActivity))
    for r in df_edges.itertuples(index=False):
        G.add_edge(int(r.mol_i), int(r.mol_j), delta=float(getattr(r, "delta_pActivity", np.nan)))

    if G.number_of_nodes() == 0:
        raise ValueError("No nodes to plot.")

    pos = nx.spring_layout(G, seed=0)
    pvals = np.array([G.nodes[n]["pActivity"] for n in G.nodes()], dtype=float)
    vmin, vmax = float(np.nanmin(pvals)), float(np.nanmax(pvals))

    plt.figure(figsize=(9, 7))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=60,
        node_color=pvals,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8)
    plt.colorbar(nodes, label="pActivity")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_series_activity(
    df_series: pd.DataFrame,
    df_pairs: pd.DataFrame,
    *,
    title: str,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = df_series.sort_values("pActivity").reset_index(drop=True)
    x = np.arange(len(df))

    idx_by_mol = {int(m): int(i) for i, m in enumerate(df["molregno"].astype(int).tolist())}

    plt.figure(figsize=(10, 4))
    plt.scatter(x, df["pActivity"].astype(float).to_numpy(), s=14)

    # Draw cliff edges as arcs between points.
    cliffs = df_pairs[df_pairs["cliff_label"] == 1]
    for r in cliffs.itertuples(index=False):
        i = idx_by_mol.get(int(r.mol_i))
        j = idx_by_mol.get(int(r.mol_j))
        if i is None or j is None:
            continue
        xi, xj = float(i), float(j)
        yi = float(df.loc[i, "pActivity"])
        yj = float(df.loc[j, "pActivity"])
        plt.plot([xi, xj], [yi, yj], alpha=0.08, color="red")

    plt.title(title)
    plt.xlabel("Compounds (ranked by pActivity within series)")
    plt.ylabel("pActivity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _highlight_diff_atoms(mol_a: Chem.Mol, mol_b: Chem.Mol) -> tuple[list[int], list[int]]:
    """
    Approximate changed atoms between two molecules by MCS.
    Returns atom indices to highlight in each molecule (non-MCS atoms).
    """
    res = rdFMCS.FindMCS(
        [mol_a, mol_b],
        completeRingsOnly=True,
        ringMatchesRingOnly=True,
        timeout=10,
    )
    if not res.smartsString:
        return list(range(mol_a.GetNumAtoms())), list(range(mol_b.GetNumAtoms()))
    patt = Chem.MolFromSmarts(res.smartsString)
    if patt is None:
        return list(range(mol_a.GetNumAtoms())), list(range(mol_b.GetNumAtoms()))

    match_a = mol_a.GetSubstructMatch(patt)
    match_b = mol_b.GetSubstructMatch(patt)
    set_a = set(match_a)
    set_b = set(match_b)
    hi_a = [i for i in range(mol_a.GetNumAtoms()) if i not in set_a]
    hi_b = [i for i in range(mol_b.GetNumAtoms()) if i not in set_b]
    return hi_a, hi_b


def draw_cliff_pair(
    smiles_a: str,
    smiles_b: str,
    *,
    legend_a: str = "A",
    legend_b: str = "B",
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        raise ValueError("Invalid SMILES provided to draw_cliff_pair.")

    hi_a, hi_b = _highlight_diff_atoms(mol_a, mol_b)
    img = Draw.MolsToGridImage(
        [mol_a, mol_b],
        legends=[legend_a, legend_b],
        molsPerRow=2,
        subImgSize=(420, 280),
        highlightAtomLists=[hi_a, hi_b],
    )
    img.save(str(outpath))

