#!/usr/bin/env python
"""
Train the final position-level HGB model on ALL data (no holdout).

Leave-one-target-out already validated generalization (NDCG@3=0.964).
This trains on all 598K rows and saves the model + metadata for the webapp.

Usage:
    python scripts/train_final_model.py
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_PATH = Path("evolve/eval_data/position_data.npz")
MODEL_DIR = Path("webapp/model")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading position data ...")
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    feature_names = list(d["feature_names"])
    print(f"  {X.shape[0]:,} rows, {X.shape[1]} features")

    # Train — same hyperparameters as the best LOO-target experiment
    print("Training HGB model (all data) ...")
    t0 = time.perf_counter()
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # Save model
    model_path = MODEL_DIR / "position_hgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model to {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")

    # Save metadata
    meta = {
        "feature_names": feature_names,
        "n_training_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets_trained_on": int(len(d["target_names"])),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_p25": float(np.percentile(y, 25)),
        "y_p75": float(np.percentile(y, 75)),
        "y_p90": float(np.percentile(y, 90)),
        "hyperparameters": {
            "max_iter": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
        },
        "validation_metrics": {
            "ndcg_at_3": 0.964,
            "hit_at_1": 0.57,
            "spearman": 0.607,
        },
    }
    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    # Feature importances (for interpretability in webapp)
    # HGB doesn't have .feature_importances_ by default in all versions,
    # but we can use permutation importance or just the built-in
    try:
        importances = model.feature_importances_
        imp_dict = dict(zip(feature_names, [float(v) for v in importances]))
        print("\n  Feature importances:")
        for name, imp in sorted(imp_dict.items(), key=lambda x: -x[1]):
            print(f"    {name:30s}  {imp:.4f}")

        meta["feature_importances"] = imp_dict
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except AttributeError:
        print("  (feature_importances_ not available)")

    print("\nDone.")


if __name__ == "__main__":
    main()
