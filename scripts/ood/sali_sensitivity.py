#!/usr/bin/env python
"""
Experiment 12: SALI-Normalized Position Sensitivity

Scientific question: Does the finding that -core_n_heavy beats ML survive when
we normalize for the SIZE of the R-group modification? Raw sensitivity
(mean |delta_p| per position) doesn't account for whether the activity change
came from a small R-group swap (genuine cliff) or a large one (trivially
expected). SALI normalization isolates positions where SMALL modifications
cause LARGE activity shifts -- the true activity cliffs.

Approach:
  1. Load raw MMP data (25M pairs) with R-group SMILES
  2. Compute R-group heavy atom counts -> modification size per MMP
  3. SALI-normalize: sali_mmp = |delta_pActivity| / modification_size
  4. Reaggregate to position level: sali_sensitivity = mean(sali_mmp)
  5. Rebuild eval data with SALI target
  6. Run full ablation: heuristics + LOO-target HGB
  7. Generate comparison figure

Usage:
    python scripts/ood/sali_sensitivity.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

MMPS_PATH = Path("outputs/mmps/all_mmps.parquet")
POSITION_DATA_PATH = Path("evolve/eval_data/position_data.npz")
OUTPUT_JSON = Path("outputs/ood/sali_sensitivity_results.json")
OUTPUT_FIG = Path("outputs/whitepaper/figure5_sali_comparison")
CHECKPOINT = Path("CHECKPOINT.md")

HGB_KWARGS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 50,
    "random_state": 42,
}


# ── Utility functions ─────────────────────────────────────────────────────────

def _n_heavy(smi: str) -> int:
    """Count heavy atoms in a SMILES string."""
    if not smi or smi == "[H]":
        return 0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    discounts = np.log2(np.arange(2, k + 2))
    pred_order = np.argsort(-y_score)[:k]
    dcg = np.sum(y_true[pred_order] / discounts)
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_order] / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def hit_rate_at_1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return float(np.argmax(y_score) == np.argmax(y_true))


def eval_metrics_for_target(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        n = mask.sum()
        if n < k:
            continue
        ndcgs.append(ndcg_at_k(y[mask], scores[mask], k))
        hits.append(hit_rate_at_1(y[mask], scores[mask]))
    if len(y) > 5:
        rho, _ = stats.spearmanr(scores, y)
    else:
        rho = 0.0
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "n_groups": len(ndcgs),
    }


def score_heuristic(score_values, y, offsets, targets, groups_arr, label):
    all_ndcgs, all_hits, all_rhos = [], [], []
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        m = eval_metrics_for_target(
            score_values[lo:hi], y[lo:hi], groups_arr[lo:hi]
        )
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])
    result = {
        "label": label,
        "type": "heuristic",
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "ndcg_per_target": {str(t): float(v) for t, v in zip(targets, all_ndcgs)},
    }
    print(f"  {label:55s}  NDCG@3={result['ndcg']:.4f} (+-{result['ndcg_std']:.4f})  "
          f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}")
    return result


def loo_target_hgb(X_in, y, offsets, targets, groups_arr, label, feat_names=None):
    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()
    for i in range(len(targets)):
        lo, hi = offsets[i], offsets[i + 1]
        X_test, y_test = X_in[lo:hi], y[lo:hi]
        g_test = groups_arr[lo:hi]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[lo:hi] = False
        X_train, y_train = X_in[train_mask], y[train_mask]
        model = HistGradientBoostingRegressor(**HGB_KWARGS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = eval_metrics_for_target(preds, y_test, g_test)
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])
        if (i + 1) % 10 == 0:
            print(f"    ... {label}: {i+1}/{len(targets)} targets done "
                  f"(running NDCG@3={np.mean(all_ndcgs):.4f})")
    elapsed = time.perf_counter() - t0
    result = {
        "label": label,
        "type": "hgb",
        "n_features": X_in.shape[1],
        "feature_names": feat_names or [],
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "ndcg_per_target": {str(t): float(v) for t, v in zip(targets, all_ndcgs)},
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  {label:55s}  NDCG@3={result['ndcg']:.4f} (+-{result['ndcg_std']:.4f})  "
          f"Hit@1={result['hit_rate']:.3f}  Spearman={result['spearman']:.4f}  "
          f"[{elapsed:.0f}s]")
    return result


def write_checkpoint(stage: str, details: str = "") -> None:
    """Write checkpoint for session recovery."""
    content = f"""# CHECKPOINT -- Experiment 12: SALI Sensitivity

**Stage:** {stage}
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## What's Done
{details}

## Resume Prompt
"Use the activity-cliffs-worker to continue Experiment 12 from CHECKPOINT.md"
"""
    CHECKPOINT.write_text(content, encoding="utf-8")
    logger.info("Checkpoint written: %s", stage)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: COMPUTE SALI-NORMALIZED MMP DATA
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_compute_sali_mmps() -> pd.DataFrame:
    """Load raw MMPs, compute R-group sizes, SALI-normalize."""
    logger.info("PHASE 1: Loading raw MMP data and computing SALI normalization")

    logger.info("  Loading %s ...", MMPS_PATH)
    mmps = pd.read_parquet(
        MMPS_PATH,
        columns=[
            "target_chembl_id", "mol_from", "core_smiles",
            "rgroup_from", "rgroup_to",
            "abs_delta_pActivity",
        ],
    )
    logger.info("  Loaded %s MMPs", f"{len(mmps):,}")

    # Compute R-group heavy atom counts
    logger.info("  Computing R-group heavy atom counts (this takes a few minutes)...")
    t0 = time.perf_counter()

    # Cache unique R-groups to avoid recomputing
    all_rgroups = set(mmps["rgroup_from"].unique()) | set(mmps["rgroup_to"].unique())
    logger.info("  %d unique R-groups to process", len(all_rgroups))

    rg_cache: dict[str, int] = {}
    for i, rg in enumerate(all_rgroups):
        rg_cache[rg] = _n_heavy(rg)
        if (i + 1) % 50000 == 0:
            logger.info("    ... %d/%d R-groups processed", i + 1, len(all_rgroups))

    elapsed_rg = time.perf_counter() - t0
    logger.info("  R-group heavy atoms computed in %.0fs", elapsed_rg)

    mmps["nheavy_from"] = mmps["rgroup_from"].map(rg_cache).astype(np.float32)
    mmps["nheavy_to"] = mmps["rgroup_to"].map(rg_cache).astype(np.float32)

    # --- SALI denominators ---

    # (a) Absolute difference in R-group heavy atoms
    mmps["delta_rg_nheavy"] = np.abs(mmps["nheavy_from"] - mmps["nheavy_to"])

    # (b) Max of the two R-group sizes (avoids 0 for isosteric swaps)
    mmps["max_rg_nheavy"] = np.maximum(mmps["nheavy_from"], mmps["nheavy_to"])

    # (c) Mean of the two R-group sizes
    mmps["mean_rg_nheavy"] = (mmps["nheavy_from"] + mmps["nheavy_to"]) / 2.0

    # --- SALI-normalized activity change ---
    # Primary: |delta_p| / (max(rgroup_size) + 1)
    # +1 epsilon avoids div/0 and stabilizes small R-groups (H, F, CH3)
    eps = 1.0
    mmps["sali_max"] = mmps["abs_delta_pActivity"] / (mmps["max_rg_nheavy"] + eps)
    mmps["sali_delta"] = mmps["abs_delta_pActivity"] / (mmps["delta_rg_nheavy"] + eps)
    mmps["sali_mean"] = mmps["abs_delta_pActivity"] / (mmps["mean_rg_nheavy"] + eps)

    # Diagnostics
    logger.info("  R-group size stats:")
    logger.info("    nheavy_from: mean=%.1f, median=%.0f, max=%.0f",
                mmps["nheavy_from"].mean(), mmps["nheavy_from"].median(),
                mmps["nheavy_from"].max())
    logger.info("    nheavy_to:   mean=%.1f, median=%.0f, max=%.0f",
                mmps["nheavy_to"].mean(), mmps["nheavy_to"].median(),
                mmps["nheavy_to"].max())
    logger.info("    delta_nheavy: mean=%.1f, median=%.0f, %% zero=%.1f%%",
                mmps["delta_rg_nheavy"].mean(), mmps["delta_rg_nheavy"].median(),
                (mmps["delta_rg_nheavy"] == 0).mean() * 100)
    logger.info("  SALI stats (max-denominator):")
    logger.info("    mean=%.3f, median=%.3f, std=%.3f",
                mmps["sali_max"].mean(), mmps["sali_max"].median(),
                mmps["sali_max"].std())

    write_checkpoint("Phase 1 complete",
                     f"- Loaded {len(mmps):,} MMPs\n"
                     f"- Computed R-group heavy atoms for {len(rg_cache):,} unique R-groups\n"
                     f"- SALI normalization computed (3 variants)")

    return mmps


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: REAGGREGATE TO POSITION LEVEL WITH SALI TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_aggregate_positions(mmps: pd.DataFrame) -> tuple:
    """Aggregate SALI-normalized MMPs to position level, rebuild eval data."""
    logger.info("PHASE 2: Aggregating to position level with SALI target")

    MIN_MMPS = 3
    MIN_POSITIONS = 3

    # Aggregate per (core_smiles, target)
    pos = (
        mmps
        .groupby(["core_smiles", "target_chembl_id"])
        .agg(
            sensitivity_original=("abs_delta_pActivity", "mean"),
            sali_max_mean=("sali_max", "mean"),
            sali_delta_mean=("sali_delta", "mean"),
            sali_mean_mean=("sali_mean", "mean"),
            sali_max_median=("sali_max", "median"),
            n_mmps=("abs_delta_pActivity", "count"),
            avg_rg_size=("max_rg_nheavy", "mean"),
        )
        .reset_index()
    )
    logger.info("  %s position-target pairs before filtering", f"{len(pos):,}")

    pos = pos[pos["n_mmps"] >= MIN_MMPS].reset_index(drop=True)
    logger.info("  %s pairs after n_mmps >= %d filter", f"{len(pos):,}", MIN_MMPS)

    # Map mol_from to positions
    mol_pos = (
        mmps[["mol_from", "core_smiles", "target_chembl_id"]]
        .drop_duplicates()
    )
    mol_pos = mol_pos.merge(pos, on=["core_smiles", "target_chembl_id"])

    # Filter: >= MIN_POSITIONS positions per (mol, target)
    pos_count = (
        mol_pos
        .groupby(["mol_from", "target_chembl_id"])
        .size()
        .reset_index(name="n_positions")
    )
    pos_count = pos_count[pos_count["n_positions"] >= MIN_POSITIONS]
    mol_pos = mol_pos.merge(
        pos_count[["mol_from", "target_chembl_id"]],
        on=["mol_from", "target_chembl_id"],
    )
    logger.info("  %s rows after position filtering (>= %d per mol)",
                f"{len(mol_pos):,}", MIN_POSITIONS)

    # Load 3D context features
    logger.info("  Loading 3D context features...")
    ctx_df = pd.read_parquet("outputs/features/context_3d.parquet")
    ctx_cols = [c for c in ctx_df.columns if c != "core_smiles"]
    ctx_lookup = dict(
        zip(ctx_df["core_smiles"], ctx_df[ctx_cols].values.astype(np.float32))
    )
    n_ctx = len(ctx_cols)

    # Core topology
    unique_cores = mol_pos["core_smiles"].unique()
    logger.info("  Computing topology for %d unique cores...", len(unique_cores))
    core_topo = {}
    for smi in unique_cores:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            nh = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)
            nr = mol.GetRingInfo().NumRings()
        else:
            nh, nr = 0, 0
        core_topo[smi] = (nh, nr)

    # Sort and build arrays
    mol_pos = mol_pos.sort_values(
        ["target_chembl_id", "mol_from", "core_smiles"]
    ).reset_index(drop=True)

    target_names = []
    target_offsets = [0]
    for target, tdf in mol_pos.groupby("target_chembl_id", sort=True):
        target_names.append(str(target))
        target_offsets.append(target_offsets[-1] + len(tdf))
    target_offsets = np.array(target_offsets)
    target_names = np.array(target_names)

    zero_ctx = np.zeros(n_ctx, dtype=np.float32)
    X_list = []
    for smi in mol_pos["core_smiles"]:
        ctx = ctx_lookup.get(smi, zero_ctx)
        nh, nr = core_topo.get(smi, (0, 0))
        row = np.concatenate([ctx, [float(nh), float(nr)]])
        X_list.append(row)

    X = np.array(X_list, dtype=np.float32)
    feature_names = [f"ctx_{c}" for c in ctx_cols] + ["core_n_heavy", "core_n_rings"]

    groups = mol_pos["mol_from"].values.astype(np.int64)
    y_original = mol_pos["sensitivity_original"].values.astype(np.float32)
    y_sali_max = mol_pos["sali_max_mean"].values.astype(np.float32)
    y_sali_delta = mol_pos["sali_delta_mean"].values.astype(np.float32)
    y_sali_mean = mol_pos["sali_mean_mean"].values.astype(np.float32)
    y_sali_max_med = mol_pos["sali_max_median"].values.astype(np.float32)
    avg_rg_sizes = mol_pos["avg_rg_size"].values.astype(np.float32)

    logger.info("  Built feature matrix: %s, %d targets", X.shape, len(target_names))

    write_checkpoint("Phase 2 complete",
                     f"- {len(mol_pos):,} positions, {len(target_names)} targets\n"
                     f"- Feature matrix: {X.shape}")

    return (X, feature_names, groups, target_offsets, target_names,
            y_original, y_sali_max, y_sali_delta, y_sali_mean,
            y_sali_max_med, avg_rg_sizes)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ABLATION STUDY ON SALI TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_ablation(X, feat_names, groups, offsets, targets, y_eval, label_suffix=""):
    """Run full ablation on a given target variable."""
    logger.info("PHASE 3: Ablation study -- %s", label_suffix)

    IDX_N_HEAVY = feat_names.index("core_n_heavy")
    CTX_COLS = [i for i, n in enumerate(feat_names) if n.startswith("ctx_")]
    TOPO_COLS = [i for i, n in enumerate(feat_names) if n.startswith("core_")]

    results = []

    print(f"\n--- Heuristic baselines ({label_suffix}) ---")

    results.append(score_heuristic(
        np.random.RandomState(42).randn(len(y_eval)),
        y_eval, offsets, targets, groups,
        "Random baseline",
    ))
    results.append(score_heuristic(
        np.full(len(y_eval), y_eval.mean()),
        y_eval, offsets, targets, groups,
        "Global mean",
    ))
    results.append(score_heuristic(
        -X[:, IDX_N_HEAVY],
        y_eval, offsets, targets, groups,
        "-core_n_heavy heuristic",
    ))

    print(f"\n--- HGB LOO-target ({label_suffix}) ---")

    results.append(loo_target_hgb(
        X[:, TOPO_COLS], y_eval, offsets, targets, groups,
        "HGB topology only (2 feat)",
        [feat_names[i] for i in TOPO_COLS],
    ))
    results.append(loo_target_hgb(
        X[:, CTX_COLS], y_eval, offsets, targets, groups,
        "HGB 3D context only (9 feat)",
        [feat_names[i] for i in CTX_COLS],
    ))
    results.append(loo_target_hgb(
        X, y_eval, offsets, targets, groups,
        "HGB full model (11 feat)",
        feat_names,
    ))

    drop_heavy = [i for i in range(X.shape[1]) if i != IDX_N_HEAVY]
    results.append(loo_target_hgb(
        X[:, drop_heavy], y_eval, offsets, targets, groups,
        "HGB full minus core_n_heavy (10 feat)",
        [feat_names[i] for i in drop_heavy],
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: COMPARISON AND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_analysis(X, feat_names, y_orig, y_sali, avg_rg_sizes):
    """Compare original vs SALI targets, analyze what changed."""
    logger.info("PHASE 4: Comparative analysis")

    IDX_N_HEAVY = feat_names.index("core_n_heavy")
    analysis = {}

    r_orig_sali, _ = stats.pearsonr(y_orig, y_sali)
    rho_orig_sali, _ = stats.spearmanr(y_orig, y_sali)
    analysis["target_correlation"] = {
        "pearson": float(r_orig_sali),
        "spearman": float(rho_orig_sali),
    }
    print(f"\n  Original vs SALI sensitivity: Pearson={r_orig_sali:.3f}, "
          f"Spearman={rho_orig_sali:.3f}")

    print("\n  Feature correlations with sensitivity:")
    print(f"  {'Feature':<25s} {'r(original)':>12s} {'r(SALI)':>12s} {'delta':>10s}")
    print("  " + "-" * 62)
    feat_corr = {}
    for j, fname in enumerate(feat_names):
        r_orig = float(np.corrcoef(X[:, j], y_orig)[0, 1])
        r_sali = float(np.corrcoef(X[:, j], y_sali)[0, 1])
        delta = r_sali - r_orig
        print(f"  {fname:<25s} {r_orig:>+12.4f} {r_sali:>+12.4f} {delta:>+10.4f}")
        feat_corr[fname] = {"r_original": r_orig, "r_sali": r_sali, "delta": delta}
    analysis["feature_correlations"] = feat_corr

    r_heavy_orig = np.corrcoef(X[:, IDX_N_HEAVY], y_orig)[0, 1]
    r_heavy_sali = np.corrcoef(X[:, IDX_N_HEAVY], y_sali)[0, 1]
    analysis["core_n_heavy_signal_change"] = {
        "r_with_original_target": float(r_heavy_orig),
        "r_with_sali_target": float(r_heavy_sali),
        "delta": float(r_heavy_sali - r_heavy_orig),
    }

    r_rg_orig = float(np.corrcoef(avg_rg_sizes, y_orig)[0, 1])
    r_rg_sali = float(np.corrcoef(avg_rg_sizes, y_sali)[0, 1])
    analysis["rgroup_size_bias"] = {
        "r_rg_size_original": r_rg_orig,
        "r_rg_size_sali": r_rg_sali,
    }
    print(f"\n  Avg R-group size vs sensitivity: original r={r_rg_orig:+.3f}, "
          f"SALI r={r_rg_sali:+.3f}")

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: GENERATE FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_figure(results_original, results_sali):
    """Generate comparison figure: original vs SALI ablation."""
    logger.info("PHASE 5: Generating comparison figure")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = ["Random baseline", "Global mean", "-core_n_heavy heuristic",
               "HGB topology only (2 feat)", "HGB 3D context only (9 feat)",
               "HGB full model (11 feat)", "HGB full minus core_n_heavy (10 feat)"]

    def get_ndcg(results_list, label):
        for r in results_list:
            if r["label"] == label:
                return r["ndcg"]
        return None

    labels_short = ["Random", "Global\nmean", "Heuristic\n(-core_n_heavy)",
                    "HGB\ntopology", "HGB\n3D only", "HGB\nfull (11)", "HGB\nno core_n_heavy"]

    ndcg_orig = [get_ndcg(results_original, c) for c in configs]
    ndcg_sali = [get_ndcg(results_sali, c) for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: grouped bar chart
    ax = axes[0]
    bars1 = ax.bar(x - width/2, ndcg_orig, width, label="Original target",
                   color="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ndcg_sali, width, label="SALI-normalized target",
                   color="#ff7f0e", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("NDCG@3", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8.5, ha="center")
    ax.set_ylim(0.83, 1.0)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("(a) Original vs SALI-normalized: full ablation",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            if height is not None:
                ax.annotate(f"{height:.3f}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha="center", va="bottom", fontsize=7)

    # Right panel: delta
    ax2 = axes[1]
    deltas = [s - o if (s is not None and o is not None) else 0
              for o, s in zip(ndcg_orig, ndcg_sali)]
    colors = ["#2ca02c" if d > 0.002 else "#d62728" if d < -0.002 else "#7f7f7f"
              for d in deltas]
    ax2.barh(x, deltas, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels_short, fontsize=8.5)
    ax2.set_xlabel("Delta NDCG@3 (SALI - original)", fontsize=10)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("(b) Change from SALI normalization",
                  fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)

    for i, d in enumerate(deltas):
        # Place labels inside the bars
        if d >= 0:
            ax2.annotate(f"{d:+.4f}", xy=(d, i),
                        xytext=(-5, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8,
                        color="white", fontweight="bold")
        else:
            ax2.annotate(f"{d:+.4f}", xy=(d, i),
                        xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8,
                        color="white", fontweight="bold")

    fig.suptitle(
        "Figure 5. SALI normalization: does controlling for modification size "
        "change the story?",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_FIG) + ".png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(str(OUTPUT_FIG) + ".svg", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("  Saved %s.png and .svg", OUTPUT_FIG)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: UPDATE PROJECT FILES
# ═══════════════════════════════════════════════════════════════════════════════

def phase6_update_status(all_results):
    """Append to STATUS.md and PROGRESS_LOG.md."""
    logger.info("PHASE 6: Updating project files")

    s = all_results["summary"]

    status_entry = f"""
## {time.strftime('%Y-%m-%d')} -- Experiment 12: SALI-Normalized Sensitivity
**Status:** Complete
**Agent:** activity-cliffs-worker (headless)
**What was done:**
- Computed SALI normalization (|delta_p| / max_rgroup_nheavy) on 25M MMPs
- Reaggregated to position level with SALI target
- Ran full ablation: heuristics + LOO-target HGB on both original and SALI targets
- Generated comparison figure (Figure 5)
**Results:**
- Original: heuristic NDCG={s['heuristic_ndcg_original']:.4f}, HGB NDCG={s['hgb_full_ndcg_original']:.4f}, gap={s['ml_vs_heuristic_gap_original']:+.4f}
- SALI: heuristic NDCG={s['heuristic_ndcg_sali']:.4f}, HGB NDCG={s['hgb_full_ndcg_sali']:.4f}, gap={s['ml_vs_heuristic_gap_sali']:+.4f}
- core_n_heavy correlation: original r={s['core_n_heavy_r_original']:+.3f}, SALI r={s['core_n_heavy_r_sali']:+.3f}
- Heuristic still wins: {s['heuristic_still_wins_sali']}
**Next:** Update white paper with SALI results, interpret implications
**Cross-project findings:** None
"""

    status_path = Path("STATUS.md")
    if status_path.exists():
        content = status_path.read_text(encoding="utf-8")
        # Insert after the header line
        lines = content.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("---"):
                insert_idx = i + 1
                break
        lines.insert(insert_idx, status_entry)
        status_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        status_path.write_text(
            "# Activity Cliffs -- Status Log\n\n---\n" + status_entry,
            encoding="utf-8",
        )

    progress_path = Path("PROGRESS_LOG.md")
    if progress_path.exists():
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n### {time.strftime('%Y-%m-%d')} -- Experiment 12: SALI Sensitivity\n")
            f.write(f"- SALI normalization computed on 25M MMPs\n")
            f.write(f"- Heuristic still wins: {s['heuristic_still_wins_sali']}\n")
            f.write(f"- ML gap changed from {s['ml_vs_heuristic_gap_original']:+.4f} "
                    f"to {s['ml_vs_heuristic_gap_sali']:+.4f}\n")

    logger.info("  STATUS.md and PROGRESS_LOG.md updated")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.perf_counter()

    print("\n" + "=" * 80)
    print("EXPERIMENT 12: SALI-NORMALIZED POSITION SENSITIVITY")
    print("=" * 80)
    print()
    print("Question: Does -core_n_heavy dominate because raw sensitivity")
    print("is biased by R-group modification size? SALI normalization")
    print("divides |delta_pActivity| by R-group size to isolate positions")
    print("where SMALL changes cause LARGE activity shifts.")
    print("=" * 80)

    # Phase 1
    mmps = phase1_compute_sali_mmps()

    # Phase 2
    (X, feat_names, groups, offsets, targets,
     y_orig, y_sali_max, y_sali_delta, y_sali_mean,
     y_sali_max_med, avg_rg_sizes) = phase2_aggregate_positions(mmps)
    del mmps

    y_sali = y_sali_max  # primary SALI target

    write_checkpoint("Phases 1-2 complete, starting ablation",
                     f"- {X.shape[0]:,} positions, {len(targets)} targets\n"
                     f"- Starting LOO-target HGB training")

    # Phase 3a: Original target
    print("\n" + "=" * 80)
    print("ABLATION ON ORIGINAL TARGET (baseline)")
    print("=" * 80)
    results_orig = phase3_ablation(
        X, feat_names, groups, offsets, targets, y_orig,
        label_suffix="original target"
    )

    write_checkpoint("Original ablation complete",
                     "- Original ablation done\n- Starting SALI ablation")

    # Phase 3b: SALI target
    print("\n" + "=" * 80)
    print("ABLATION ON SALI-NORMALIZED TARGET")
    print("=" * 80)
    results_sali = phase3_ablation(
        X, feat_names, groups, offsets, targets, y_sali,
        label_suffix="SALI target"
    )

    # Phase 4
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    analysis = phase4_analysis(X, feat_names, y_orig, y_sali, avg_rg_sizes)

    # Phase 5
    phase5_figure(results_orig, results_sali)

    # Build final results
    def get_metric(results_list, label, metric="ndcg"):
        for r in results_list:
            if r["label"] == label:
                return r[metric]
        return None

    heur_orig = get_metric(results_orig, "-core_n_heavy heuristic")
    heur_sali = get_metric(results_sali, "-core_n_heavy heuristic")
    full_orig = get_metric(results_orig, "HGB full model (11 feat)")
    full_sali = get_metric(results_sali, "HGB full model (11 feat)")

    all_results = {
        "experiment": "Experiment 12: SALI-normalized position sensitivity",
        "date": time.strftime("%Y-%m-%d"),
        "sali_definition": "|delta_pActivity| / (max_rgroup_n_heavy + 1)",
        "n_positions": int(X.shape[0]),
        "n_targets": int(len(targets)),
        "results_original_target": results_orig,
        "results_sali_target": results_sali,
        "analysis": analysis,
        "summary": {
            "heuristic_still_wins_sali": bool(heur_sali > full_sali) if (heur_sali and full_sali) else None,
            "heuristic_ndcg_original": heur_orig,
            "heuristic_ndcg_sali": heur_sali,
            "hgb_full_ndcg_original": full_orig,
            "hgb_full_ndcg_sali": full_sali,
            "ml_vs_heuristic_gap_original": (full_orig - heur_orig) if (full_orig and heur_orig) else None,
            "ml_vs_heuristic_gap_sali": (full_sali - heur_sali) if (full_sali and heur_sali) else None,
            "core_n_heavy_r_original": analysis["core_n_heavy_signal_change"]["r_with_original_target"],
            "core_n_heavy_r_sali": analysis["core_n_heavy_signal_change"]["r_with_sali_target"],
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", OUTPUT_JSON)

    # Phase 6
    phase6_update_status(all_results)

    # Print summary
    elapsed_total = time.perf_counter() - t_global
    s = all_results["summary"]
    print("\n" + "=" * 80)
    print("EXPERIMENT 12 SUMMARY")
    print("=" * 80)
    print(f"\n  Original target:")
    print(f"    Heuristic NDCG@3:  {s['heuristic_ndcg_original']:.4f}")
    print(f"    HGB full NDCG@3:   {s['hgb_full_ndcg_original']:.4f}")
    print(f"    Gap (ML - heur):   {s['ml_vs_heuristic_gap_original']:+.4f}")
    print(f"\n  SALI-normalized target:")
    print(f"    Heuristic NDCG@3:  {s['heuristic_ndcg_sali']:.4f}")
    print(f"    HGB full NDCG@3:   {s['hgb_full_ndcg_sali']:.4f}")
    print(f"    Gap (ML - heur):   {s['ml_vs_heuristic_gap_sali']:+.4f}")
    print(f"\n  core_n_heavy correlation:")
    print(f"    With original:     {s['core_n_heavy_r_original']:+.4f}")
    print(f"    With SALI:         {s['core_n_heavy_r_sali']:+.4f}")

    if s["heuristic_still_wins_sali"]:
        print("\n  VERDICT: Heuristic STILL dominates under SALI normalization.")
        print("  The scaffold-size signal is a genuine physical principle,")
        print("  not an artifact of R-group size bias.")
    else:
        print("\n  VERDICT: ML NOW BEATS the heuristic under SALI normalization!")
        print("  The original dominance of -core_n_heavy was partly an artifact.")
        print("  3D context and other features genuinely matter for predicting")
        print("  where SMALL modifications cause LARGE activity shifts.")

    print(f"\n  Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print("=" * 80)

    # Clean up checkpoint on success
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        logger.info("Checkpoint removed (experiment complete)")


if __name__ == "__main__":
    main()
