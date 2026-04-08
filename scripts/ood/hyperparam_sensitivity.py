#!/usr/bin/env python
"""
OOD Experiment 11: Hyperparameter Sensitivity

Random search over 50 HGB configurations to demonstrate that the
reported NDCG@3 is robust to hyperparameter choices.

Protocol:
  - Random search using loguniform/randint distributions (scipy.stats)
  - Parameters: learning_rate, max_depth, max_iter, min_samples_leaf,
    max_leaf_nodes, l2_regularization
  - LOO-target NDCG@3 for each configuration (same protocol as main experiments)
  - Report distribution: mean, std, min, max of NDCG@3 across 50 configs
  - Shows range is <0.02 (or reports honestly if it isn't)
  - Saves results as JSON and strip plot / box plot as PNG

Usage:
    python scripts/ood/hyperparam_sensitivity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.stdout.reconfigure(line_buffering=True)

# ── Load data ────────────────────────────────────────────────────────────────

EVAL_DATA_PATH = Path("evolve/eval_data/position_data.npz")
_data = np.load(EVAL_DATA_PATH, allow_pickle=True)
X = _data["X"]
Y = _data["y"]
GROUPS = _data["groups"]
OFFSETS = _data["target_offsets"]
TARGETS = _data["target_names"]
FEAT_NAMES = list(_data["feature_names"])
del _data

print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features, {len(TARGETS)} targets")

IDX_N_HEAVY = FEAT_NAMES.index("core_n_heavy")


# ── Evaluation functions ─────────────────────────────────────────────────────

def ndcg_at_k(y_true, y_score, k=3):
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


def hit_rate_at_1(y_true, y_score):
    if len(y_true) < 2:
        return 0.0
    return float(np.argmax(y_score) == np.argmax(y_true))


def eval_metrics_for_target(scores, y, groups, k=3):
    ndcgs, hits = [], []
    for mol_id in np.unique(groups):
        mask = groups == mol_id
        if mask.sum() < k:
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
    }


# ── Hyperparameter search space (continuous distributions) ───────────────────

# scipy.stats distributions for random search
# loguniform(a, b) samples uniformly in log-space between a and b
# randint(lo, hi) samples integers in [lo, hi)
PARAM_DISTRIBUTIONS = {
    "learning_rate": stats.loguniform(0.01, 0.3),    # loguniform [0.01, 0.30]
    "max_depth": stats.randint(3, 15),                # int [3, 14]
    "max_iter": stats.randint(50, 500),               # int [50, 499]
    "min_samples_leaf": stats.randint(10, 100),       # int [10, 99]
    "max_leaf_nodes": stats.randint(15, 127),         # int [15, 126]
    "l2_regularization": stats.loguniform(0.001, 10.0),  # loguniform [0.001, 10.0]
}


def sample_config(rng_seed):
    """Sample one random configuration from the search space."""
    rng = np.random.RandomState(rng_seed)
    config = {}
    for param, dist in PARAM_DISTRIBUTIONS.items():
        val = dist.rvs(random_state=rng)
        if hasattr(dist, "a") and isinstance(dist.a, int) or param in ("max_depth", "max_iter",
                                                                         "min_samples_leaf",
                                                                         "max_leaf_nodes"):
            config[param] = int(val)
        else:
            config[param] = float(val)
    return config


def loo_target_hgb(hgb_kwargs, label):
    """Run LOO-target evaluation with given HGB hyperparameters."""
    all_ndcgs, all_hits, all_rhos = [], [], []
    t0 = time.perf_counter()

    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        train_mask = np.ones(len(Y), dtype=bool)
        train_mask[lo:hi] = False

        model = HistGradientBoostingRegressor(**hgb_kwargs)
        model.fit(X[train_mask], Y[train_mask])
        preds = model.predict(X[lo:hi])

        m = eval_metrics_for_target(preds, Y[lo:hi], GROUPS[lo:hi])
        all_ndcgs.append(m["ndcg"])
        all_hits.append(m["hit_rate"])
        all_rhos.append(m["spearman"])

    elapsed = time.perf_counter() - t0
    result = {
        "label": label,
        "params": {k: v for k, v in hgb_kwargs.items() if k != "random_state"},
        "ndcg": float(np.mean(all_ndcgs)),
        "hit_rate": float(np.mean(all_hits)),
        "spearman": float(np.mean(all_rhos)),
        "ndcg_std": float(np.std(all_ndcgs)),
        "per_target_ndcg": [round(v, 5) for v in all_ndcgs],
        "elapsed_s": round(elapsed, 1),
    }
    lr = hgb_kwargs.get("learning_rate", "?")
    md = hgb_kwargs.get("max_depth", "?")
    mi = hgb_kwargs.get("max_iter", "?")
    print(f"  {label:40s}  NDCG@3={result['ndcg']:.4f} (+-{result['ndcg_std']:.4f})"
          f"  Spearman={result['spearman']:.4f}  [{elapsed:.0f}s]")
    return result


def make_plots(all_results, heur_ndcg, out_dir):
    """Generate strip plot and box plot of NDCG@3 distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return None

    ndcgs = np.array([r["ndcg"] for r in all_results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Strip plot (left) ---
    ax = axes[0]
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.15, 0.15, len(ndcgs))
    ax.scatter(np.ones(len(ndcgs)) + jitter, ndcgs,
               alpha=0.7, s=40, color="steelblue", zorder=3)
    # Mark default config (index 0)
    ax.scatter([1.0], [all_results[0]["ndcg"]],
               s=120, color="darkorange", marker="D", zorder=5, label="Default config")
    # Heuristic reference line
    ax.axhline(heur_ndcg, color="crimson", linewidth=1.5, linestyle="--",
               label=f"-core_n_heavy = {heur_ndcg:.4f}")
    # Range band
    ax.axhspan(np.min(ndcgs), np.max(ndcgs), alpha=0.08, color="steelblue")

    ax.set_xlim(0.6, 1.4)
    ax.set_xticks([])
    ax.set_ylabel("NDCG@3", fontsize=12)
    ax.set_title("Strip Plot: NDCG@3 across 51 HGB configs", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotation
    ndcg_range = float(np.max(ndcgs) - np.min(ndcgs))
    label_text = (f"mean={np.mean(ndcgs):.4f}\n"
                  f"std={np.std(ndcgs):.4f}\n"
                  f"range={ndcg_range:.4f}")
    ax.text(0.72, np.min(ndcgs) + ndcg_range * 0.05, label_text,
            fontsize=8, va="bottom", color="navy",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- Box plot (right) ---
    ax = axes[1]
    bp = ax.boxplot([ndcgs], positions=[1], widths=0.5,
                    patch_artist=True, notch=False,
                    boxprops=dict(facecolor="steelblue", alpha=0.5),
                    medianprops=dict(color="navy", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=6, alpha=0.6))

    # Overlay points
    rng2 = np.random.RandomState(1)
    jitter2 = rng2.uniform(-0.12, 0.12, len(ndcgs))
    ax.scatter(np.ones(len(ndcgs)) + jitter2, ndcgs,
               alpha=0.5, s=25, color="steelblue", zorder=3)
    ax.scatter([1.0], [all_results[0]["ndcg"]],
               s=100, color="darkorange", marker="D", zorder=5, label="Default config")
    ax.axhline(heur_ndcg, color="crimson", linewidth=1.5, linestyle="--",
               label=f"-core_n_heavy = {heur_ndcg:.4f}")

    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels(["HGB configs"])
    ax.set_ylabel("NDCG@3", fontsize=12)
    ax.set_title("Box Plot: NDCG@3 across 51 HGB configs", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Robustness annotation
    if ndcg_range < 0.01:
        verdict = "VERY ROBUST (range < 0.01)"
        color = "green"
    elif ndcg_range < 0.02:
        verdict = "ROBUST (range < 0.02)"
        color = "green"
    elif ndcg_range < 0.05:
        verdict = "MODERATELY ROBUST"
        color = "orange"
    else:
        verdict = "SENSITIVE (range >= 0.05)"
        color = "red"

    fig.suptitle(
        f"Hyperparameter Sensitivity: {verdict}",
        fontsize=13, fontweight="bold", color=color, y=1.01
    )

    plt.tight_layout()
    plot_path = out_dir / "hyperparam_sensitivity_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {plot_path}")
    return plot_path


def main():
    print("\n" + "=" * 78)
    print("OOD EXPERIMENT 11: HYPERPARAMETER SENSITIVITY")
    print("=" * 78)
    print("\nSearch space (continuous distributions):")
    print("  learning_rate    : loguniform(0.01, 0.30)")
    print("  max_depth        : randint(3, 15)")
    print("  max_iter         : randint(50, 500)")
    print("  min_samples_leaf : randint(10, 100)")
    print("  max_leaf_nodes   : randint(15, 127)")
    print("  l2_regularization: loguniform(0.001, 10.0)")
    print()

    # Heuristic baseline for reference
    print("--- Heuristic baseline ---")
    heur_ndcgs = []
    for i in range(len(TARGETS)):
        lo, hi = OFFSETS[i], OFFSETS[i + 1]
        scores = -X[lo:hi, IDX_N_HEAVY]
        m = eval_metrics_for_target(scores, Y[lo:hi], GROUPS[lo:hi])
        heur_ndcgs.append(m["ndcg"])
    heur_ndcg = float(np.mean(heur_ndcgs))
    print(f"  -core_n_heavy heuristic:  NDCG@3={heur_ndcg:.4f}\n")

    # Default configuration (matching main experiments)
    print("--- Default HGB configuration ---")
    default_kwargs = {
        "max_iter": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_samples_leaf": 50,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "random_state": 42,
    }
    default_result = loo_target_hgb(default_kwargs, "Default (300/6/0.10/50/31/0.0)")

    # Random search with continuous distributions
    n_configs = 50
    print(f"\n--- Random search ({n_configs} configurations) ---")

    all_results = [default_result]
    for ci in range(n_configs):
        # Use ci as seed so results are reproducible
        config = sample_config(rng_seed=ci + 100)
        label = (f"Config {ci+1:02d} "
                 f"(lr={config['learning_rate']:.3f} "
                 f"d={config['max_depth']} "
                 f"iter={config['max_iter']})")
        hgb_kwargs = {**config, "random_state": 42}
        result = loo_target_hgb(hgb_kwargs, label)
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    ndcgs = np.array([r["ndcg"] for r in all_results])
    spearman_vals = np.array([r["spearman"] for r in all_results])
    ndcg_range = float(np.max(ndcgs) - np.min(ndcgs))
    ndcg_std = float(np.std(ndcgs))

    print("\n" + "=" * 78)
    print("HYPERPARAMETER SENSITIVITY SUMMARY")
    print("=" * 78)

    print(f"\n  Configurations tested: {len(all_results)} "
          f"(1 default + {n_configs} random)")

    print(f"\n  NDCG@3 distribution across {len(all_results)} configs:")
    print(f"    mean   = {np.mean(ndcgs):.4f}")
    print(f"    std    = {ndcg_std:.4f}")
    print(f"    min    = {np.min(ndcgs):.4f}")
    print(f"    p25    = {np.percentile(ndcgs, 25):.4f}")
    print(f"    median = {np.median(ndcgs):.4f}")
    print(f"    p75    = {np.percentile(ndcgs, 75):.4f}")
    print(f"    max    = {np.max(ndcgs):.4f}")
    print(f"    range  = {ndcg_range:.4f}  <-- KEY METRIC (target: <0.02)")

    print(f"\n  Spearman distribution:")
    print(f"    mean   = {np.mean(spearman_vals):.4f}")
    print(f"    std    = {np.std(spearman_vals):.4f}")
    print(f"    range  = {np.max(spearman_vals) - np.min(spearman_vals):.4f}")

    n_beat_heur = int(np.sum(ndcgs > heur_ndcg))
    print(f"\n  -core_n_heavy heuristic NDCG@3: {heur_ndcg:.4f}")
    print(f"  Configs beating heuristic: {n_beat_heur}/{len(all_results)}")

    # Top/bottom configs
    sorted_results = sorted(all_results, key=lambda r: r["ndcg"], reverse=True)
    print(f"\n  Top 5 configurations:")
    for r in sorted_results[:5]:
        print(f"    NDCG@3={r['ndcg']:.4f}  Spearman={r['spearman']:.4f}  {r['params']}")
    print(f"\n  Bottom 5 configurations:")
    for r in sorted_results[-5:]:
        print(f"    NDCG@3={r['ndcg']:.4f}  Spearman={r['spearman']:.4f}  {r['params']}")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if ndcg_range < 0.01:
        verdict = "VERY ROBUST"
        print(f"  -> VERY ROBUST: NDCG@3 range={ndcg_range:.4f} (< 0.01)")
    elif ndcg_range < 0.02:
        verdict = "ROBUST"
        print(f"  -> ROBUST: NDCG@3 range={ndcg_range:.4f} (< 0.02)")
    elif ndcg_range < 0.05:
        verdict = "MODERATELY ROBUST"
        print(f"  -> MODERATELY ROBUST: NDCG@3 range={ndcg_range:.4f}")
    else:
        verdict = "SENSITIVE"
        print(f"  -> SENSITIVE: NDCG@3 range={ndcg_range:.4f} (>= 0.05)")
        print(f"     Some configs underperform substantially")

    if n_beat_heur == 0:
        print(f"  -> NO config beats the heuristic -- confirms core_n_heavy dominance")

    # ── Generate plots ────────────────────────────────────────────────────
    out_dir = Path("outputs/ood")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Generating plots ---")
    make_plots(all_results, heur_ndcg, out_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = out_dir / "hyperparam_sensitivity_results.json"

    # Strip per_target_ndcg from stored results to keep file manageable
    results_for_json = []
    for r in all_results:
        r2 = {k: v for k, v in r.items() if k != "per_target_ndcg"}
        results_for_json.append(r2)

    output = {
        "experiment": "OOD Experiment 11: Hyperparameter Sensitivity",
        "search_space": {
            "learning_rate": "loguniform(0.01, 0.30)",
            "max_depth": "randint(3, 15)",
            "max_iter": "randint(50, 500)",
            "min_samples_leaf": "randint(10, 100)",
            "max_leaf_nodes": "randint(15, 127)",
            "l2_regularization": "loguniform(0.001, 10.0)",
        },
        "heuristic_ndcg": heur_ndcg,
        "results": results_for_json,
        "summary": {
            "n_configs": len(all_results),
            "ndcg_mean": float(np.mean(ndcgs)),
            "ndcg_std": ndcg_std,
            "ndcg_min": float(np.min(ndcgs)),
            "ndcg_p25": float(np.percentile(ndcgs, 25)),
            "ndcg_median": float(np.median(ndcgs)),
            "ndcg_p75": float(np.percentile(ndcgs, 75)),
            "ndcg_max": float(np.max(ndcgs)),
            "ndcg_range": ndcg_range,
            "spearman_mean": float(np.mean(spearman_vals)),
            "spearman_std": float(np.std(spearman_vals)),
            "n_beat_heuristic": n_beat_heur,
            "verdict": verdict,
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
