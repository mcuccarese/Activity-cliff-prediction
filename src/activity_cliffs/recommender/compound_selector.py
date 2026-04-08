"""
Joint Compound Selector: Synthesizability-aware diverse compound recommendation.

Replaces the two-stage approach (pick axes, then find compounds) with a single
optimization that selects k compounds jointly maximizing:
  1. Coverage: diversity in property-change space (non-redundant hypotheses)
  2. Synthesizability: low SA score (easy to make)
  3. Impact: historical activity change magnitude

Operates directly on the available MMP transforms at a position, not on
abstract change-type axes. Each transform is a real compound with known
properties — no abstraction layer needed.

Algorithm: Greedy submodular maximization with composite scoring.
At each step, the next compound is chosen to maximize:
    score = coverage^alpha * synthesizability^beta * impact^gamma
where:
    coverage = min distance to any already-selected compound in standardized
               property-delta space (higher = more novel)
    synthesizability = 1 / SA_score (higher = easier to make)
    impact = historical |delta_pActivity| (higher = more informative)

Greedy selection achieves a (1 - 1/e) approximation guarantee for monotone
submodular objectives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class ScoredCompound:
    """A compound candidate with all scoring components."""
    smiles: str
    rgroup_from: str
    rgroup_to: str
    core_smiles: str
    delta_props: np.ndarray          # 11D standardized property change vector
    delta_props_raw: dict[str, float] # human-readable property changes
    sa_score: float                   # 1-10, lower = easier
    historical_impact: float          # mean |delta_pActivity| from MMP corpus
    mol_weight: float
    # Scores assigned during selection
    coverage_score: float = 0.0
    synth_score: float = 0.0
    impact_score: float = 0.0
    composite_score: float = 0.0
    rank: int = 0
    dominant_change: str = ""         # which property changes most


@dataclass
class CompoundRecommendation:
    """The output: k recommended compounds with full reasoning."""
    position_core: str
    input_smiles: str
    n_candidates: int
    n_after_sa_filter: int
    selected: list[ScoredCompound]
    # Coverage analysis
    pairwise_distances: list[float]   # distances between selected compounds
    property_coverage: float          # fraction of property space spanned
    # Parameters used
    alpha: float
    beta: float
    gamma: float
    sa_threshold: float


@dataclass
class AdaptiveRecommendation:
    """Output of adaptive_select: a CompoundRecommendation plus scaffold-similarity metadata."""
    recommendation: CompoundRecommendation
    query_core: str
    max_tanimoto: float               # best Tanimoto to any training core
    regime: str                        # "novel", "familiar", or "interpolated"
    alpha_used: float                  # coverage exponent actually used
    gamma_used: float                  # impact exponent actually used


class CompoundSelector:
    """
    Jointly selects k synthesizable, diverse, high-impact compounds.

    Parameters
    ----------
    change_type_names : list[str]
        Names of the 11 property-change axes.
    sigmas : dict[str, float]
        Per-axis standard deviations for standardization.
    alpha, beta, gamma : float
        Exponents for coverage, synthesizability, and impact in the
        composite score. Higher = more weight on that objective.
    sa_threshold : float
        Maximum SA score to consider (1-10). Default 5.0.
    """

    def __init__(
        self,
        change_type_names: list[str] | None = None,
        sigmas: dict[str, float] | None = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        sa_threshold: float = 5.0,
    ):
        import json
        from activity_cliffs.features.change_type import CHANGE_TYPE_NAMES, RGROUP_PROP_NAMES

        self.change_type_names = change_type_names or CHANGE_TYPE_NAMES
        self.rgroup_prop_names = RGROUP_PROP_NAMES

        if sigmas is None:
            meta_path = ROOT / "webapp" / "model" / "change_type_meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.sigmas = meta["delta_prop_sigmas"]
        else:
            self.sigmas = sigmas

        self.sigma_vec = np.array(
            [max(self.sigmas.get(c, 1.0), 1e-6) for c in self.change_type_names],
            dtype=np.float32,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sa_threshold = sa_threshold

        # Lazy-loaded
        self._rgroup_props: dict[str, np.ndarray] | None = None

    def _load_rgroup_props(self) -> dict[str, np.ndarray]:
        if self._rgroup_props is None:
            rg_path = ROOT / "outputs" / "features" / "rgroup_props.parquet"
            df = pd.read_parquet(rg_path)
            self._rgroup_props = dict(
                zip(df["rgroup_smiles"], df[self.rgroup_prop_names].values.astype(np.float32))
            )
        return self._rgroup_props

    def _sa_score(self, smiles: str) -> float:
        """Synthetic accessibility score (1=easy, 10=hard)."""
        try:
            from rdkit.Chem import RDConfig
            import sys, os
            sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            import sascorer
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0
            return sascorer.calculateScore(mol)
        except Exception:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0
            from rdkit.Chem import rdMolDescriptors
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            mw = Descriptors.MolWt(mol)
            return min(10.0, 1.0 + n_rings * 0.5 + n_chiral * 1.0 + mw / 500.0)

    def _build_candidates(self, mmps: pd.DataFrame) -> list[ScoredCompound]:
        """Convert raw MMPs into scored candidate compounds."""
        rg_props = self._load_rgroup_props()
        zero_prop = np.zeros(len(self.rgroup_prop_names), dtype=np.float32)

        candidates = []
        seen_rgroups = set()

        for _, row in mmps.iterrows():
            rg_to = row["rgroup_to"]
            if rg_to in seen_rgroups:
                continue

            pf = rg_props.get(row["rgroup_from"])
            pt = rg_props.get(rg_to)
            if pf is None or pt is None:
                continue

            delta_raw = pt - pf
            delta_std = delta_raw / self.sigma_vec

            product_smiles = row["smiles_to"]
            sa = self._sa_score(product_smiles)

            if sa > self.sa_threshold:
                continue

            mol = Chem.MolFromSmiles(product_smiles)
            mw = Descriptors.MolWt(mol) if mol else 0.0

            # Identify dominant property change
            dominant_idx = int(np.argmax(np.abs(delta_std)))
            dominant_name = self.change_type_names[dominant_idx]

            delta_dict = {
                name: float(delta_raw[j])
                for j, name in enumerate(self.change_type_names)
            }

            candidates.append(ScoredCompound(
                smiles=product_smiles,
                rgroup_from=row["rgroup_from"],
                rgroup_to=rg_to,
                core_smiles=row["core_smiles"],
                delta_props=delta_std,
                delta_props_raw=delta_dict,
                sa_score=sa,
                historical_impact=float(row["abs_delta_pActivity"]),
                mol_weight=mw,
                dominant_change=dominant_name,
            ))
            seen_rgroups.add(rg_to)

        return candidates

    def select(
        self,
        mol_smiles: str,
        core_smiles: str,
        k: int = 3,
        mmps: pd.DataFrame | None = None,
    ) -> CompoundRecommendation:
        """
        Select k compounds that jointly maximize coverage, synthesizability,
        and impact at a given position.

        Parameters
        ----------
        mol_smiles : str
            Input molecule SMILES.
        core_smiles : str
            Core SMILES defining the position (with [*:1]).
        k : int
            Number of compounds to select.
        mmps : pd.DataFrame, optional
            Pre-loaded MMPs for this position. If None, loads from disk.

        Returns
        -------
        CompoundRecommendation
        """
        # Load MMPs if not provided
        if mmps is None:
            mmps_path = ROOT / "outputs" / "mmps" / "all_mmps.parquet"
            mmps = pd.read_parquet(
                mmps_path,
                columns=["target_chembl_id", "core_smiles", "smiles_from", "smiles_to",
                         "rgroup_from", "rgroup_to", "abs_delta_pActivity"],
                filters=[("core_smiles", "==", core_smiles)],
            )

        n_total = len(mmps)
        logger.info("Position %s: %d MMPs", core_smiles[:40], n_total)

        # Build candidates (deduped by R-group, SA-filtered)
        candidates = self._build_candidates(mmps)
        n_filtered = len(candidates)
        logger.info("  %d candidates after SA filter (<= %.1f)", n_filtered, self.sa_threshold)

        if n_filtered == 0:
            return CompoundRecommendation(
                position_core=core_smiles,
                input_smiles=mol_smiles,
                n_candidates=n_total,
                n_after_sa_filter=0,
                selected=[],
                pairwise_distances=[],
                property_coverage=0.0,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                sa_threshold=self.sa_threshold,
            )

        # Normalize scoring components to [0, 1]
        impacts = np.array([c.historical_impact for c in candidates])
        sa_scores = np.array([c.sa_score for c in candidates])

        # Synthesizability: invert and normalize (lower SA = higher score)
        synth_scores = 1.0 / sa_scores
        if synth_scores.max() > synth_scores.min():
            synth_norm = (synth_scores - synth_scores.min()) / (synth_scores.max() - synth_scores.min())
        else:
            synth_norm = np.ones_like(synth_scores)

        # Impact: normalize
        if impacts.max() > impacts.min():
            impact_norm = (impacts - impacts.min()) / (impacts.max() - impacts.min())
        else:
            impact_norm = np.ones_like(impacts)

        # Property delta matrix for coverage computation
        delta_matrix = np.array([c.delta_props for c in candidates])

        # ── Greedy submodular selection ───────────────────────────────────
        selected_idx: list[int] = []

        for step in range(min(k, n_filtered)):
            best_idx = -1
            best_score = -np.inf

            for i in range(n_filtered):
                if i in selected_idx:
                    continue

                # Coverage: min distance to any already-selected compound
                if selected_idx:
                    dists = [
                        float(np.linalg.norm(delta_matrix[i] - delta_matrix[j]))
                        for j in selected_idx
                    ]
                    coverage = min(dists)
                else:
                    # First compound: coverage = distance from origin
                    coverage = float(np.linalg.norm(delta_matrix[i]))

                # Normalize coverage by max possible (diagonal of property space)
                max_coverage = float(np.sqrt(len(self.change_type_names)))
                coverage_norm = min(coverage / max_coverage, 1.0)

                # Composite score
                score = (
                    coverage_norm ** self.alpha
                    * synth_norm[i] ** self.beta
                    * impact_norm[i] ** self.gamma
                )

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                selected_idx.append(best_idx)

                # Store component scores
                candidates[best_idx].rank = step + 1
                candidates[best_idx].coverage_score = coverage_norm if step > 0 else float(np.linalg.norm(delta_matrix[best_idx])) / max_coverage
                candidates[best_idx].synth_score = float(synth_norm[best_idx])
                candidates[best_idx].impact_score = float(impact_norm[best_idx])
                candidates[best_idx].composite_score = float(best_score)

        selected = [candidates[i] for i in selected_idx]

        # Pairwise distances between selected compounds
        pairwise = []
        for i in range(len(selected_idx)):
            for j in range(i + 1, len(selected_idx)):
                d = float(np.linalg.norm(
                    delta_matrix[selected_idx[i]] - delta_matrix[selected_idx[j]]
                ))
                pairwise.append(d)

        # Property coverage: volume of the simplex spanned by selected deltas
        if len(selected_idx) >= 2:
            sel_deltas = delta_matrix[selected_idx]
            # Use sum of pairwise distances as a coverage proxy
            total_dist = sum(pairwise) if pairwise else 0.0
            # Normalize by what random selection would give (estimated)
            all_pairwise = []
            rng = np.random.RandomState(42)
            for _ in range(100):
                rand_idx = rng.choice(n_filtered, size=min(k, n_filtered), replace=False)
                rand_dist = 0.0
                for a in range(len(rand_idx)):
                    for b in range(a + 1, len(rand_idx)):
                        rand_dist += np.linalg.norm(delta_matrix[rand_idx[a]] - delta_matrix[rand_idx[b]])
                all_pairwise.append(rand_dist)
            mean_random = np.mean(all_pairwise)
            coverage = total_dist / mean_random if mean_random > 0 else 1.0
        else:
            coverage = 0.0

        return CompoundRecommendation(
            position_core=core_smiles,
            input_smiles=mol_smiles,
            n_candidates=n_total,
            n_after_sa_filter=n_filtered,
            selected=selected,
            pairwise_distances=pairwise,
            property_coverage=coverage,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            sa_threshold=self.sa_threshold,
        )

    def format_recommendation(self, rec: CompoundRecommendation) -> str:
        """Format a recommendation as human-readable text."""
        from activity_cliffs.recommender.hypothesis_selector import AXIS_LABELS

        lines = []
        lines.append(f"Position: {rec.position_core[:60]}")
        lines.append(f"Candidates: {rec.n_after_sa_filter} synthesizable "
                     f"(SA <= {rec.sa_threshold}) from {rec.n_candidates} total MMPs")
        lines.append(f"Coverage vs random: {rec.property_coverage:.2f}x "
                     f"(>1 = better than random diversity)")
        lines.append(f"Weights: coverage^{rec.alpha} * synth^{rec.beta} * impact^{rec.gamma}")
        lines.append("")

        if not rec.selected:
            lines.append("  No synthesizable compounds found at this position.")
            return "\n".join(lines)

        for cpd in rec.selected:
            dominant_label = AXIS_LABELS.get(cpd.dominant_change, cpd.dominant_change)
            lines.append(f"  Compound {cpd.rank}: {cpd.smiles}")
            lines.append(f"    R-group: {cpd.rgroup_from} --> {cpd.rgroup_to}")
            lines.append(f"    Dominant change: {dominant_label}")
            lines.append(f"    SA score: {cpd.sa_score:.1f}  |  Impact: {cpd.historical_impact:.2f}  |  MW: {cpd.mol_weight:.0f}")
            lines.append(f"    Scores: coverage={cpd.coverage_score:.2f}  synth={cpd.synth_score:.2f}  impact={cpd.impact_score:.2f}  composite={cpd.composite_score:.3f}")

            # Show top 3 property changes
            sorted_deltas = sorted(cpd.delta_props_raw.items(), key=lambda x: -abs(x[1]))
            top_changes = [(AXIS_LABELS.get(k, k), v) for k, v in sorted_deltas[:3]]
            changes_str = ", ".join(f"{label}: {val:+.2f}" for label, val in top_changes)
            lines.append(f"    Key changes: {changes_str}")
            lines.append("")

        # Diversity summary
        if len(rec.selected) >= 2:
            lines.append(f"  Pairwise distances: {', '.join(f'{d:.2f}' for d in rec.pairwise_distances)}")
            dominant_set = set(c.dominant_change for c in rec.selected)
            dom_labels = [AXIS_LABELS.get(d, d) for d in dominant_set]
            lines.append(f"  Distinct change families covered: {len(dominant_set)} ({', '.join(dom_labels)})")

        return "\n".join(lines)

    # ── Adaptive selector (Experiment 16 insight) ─────────────────────
    # On novel scaffolds, the impact model doesn't transfer, so diversity-only
    # selection wins.  On familiar scaffolds, impact adds genuine signal.
    # adaptive_select interpolates between the two regimes based on scaffold
    # similarity to the training corpus.

    def _load_training_core_fps(self) -> list:
        """Load and cache Morgan fingerprints (radius 2) for all unique training cores.

        Reads unique core_smiles from outputs/mmps/all_mmps.parquet, converts
        each to a Morgan fingerprint, and caches the result.  ~104K cores;
        takes ~1 min on first call, then instant.
        """
        if hasattr(self, "_training_core_fps") and self._training_core_fps is not None:
            return self._training_core_fps

        logger.info("Loading training core fingerprints (first call only)...")
        mmps_path = ROOT / "outputs" / "mmps" / "all_mmps.parquet"
        cores_df = pd.read_parquet(mmps_path, columns=["core_smiles"])
        unique_cores = cores_df["core_smiles"].unique()
        logger.info("  %d unique cores to fingerprint", len(unique_cores))

        fps = []
        n_failed = 0
        for smi in unique_cores:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_failed += 1
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)

        logger.info("  %d fingerprints computed (%d failed to parse)", len(fps), n_failed)
        self._training_core_fps = fps
        return fps

    def _max_tanimoto_to_training(self, core_smiles: str) -> float:
        """Compute the maximum Tanimoto similarity between a query core and all training cores."""
        fps = self._load_training_core_fps()

        mol = Chem.MolFromSmiles(core_smiles)
        if mol is None:
            logger.warning("Could not parse query core: %s", core_smiles)
            return 0.0

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
        return float(max(similarities)) if similarities else 0.0

    def adaptive_select(
        self,
        mol_smiles: str,
        core_smiles: str,
        k: int = 3,
        mmps: pd.DataFrame | None = None,
        *,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> AdaptiveRecommendation:
        """Select compounds using scaffold-similarity-adaptive weighting.

        Based on Experiment 16 finding: on novel scaffolds (low Tanimoto to
        training data), diversity-only selection outperforms the joint
        selector because the impact model doesn't transfer.  On familiar
        scaffolds, the impact model adds genuine signal.

        Parameters
        ----------
        mol_smiles, core_smiles, k, mmps
            Same as ``select()``.
        low_threshold : float
            Max Tanimoto below which we use diversity-only (default 0.3).
        high_threshold : float
            Max Tanimoto above which we use full impact weighting (default 0.7).

        Returns
        -------
        AdaptiveRecommendation
            Contains the CompoundRecommendation plus similarity metadata.
        """
        max_sim = self._max_tanimoto_to_training(core_smiles)

        # ── Determine regime and interpolate weights ──────────────────
        # Novel:    alpha=2.0, gamma=0.0  (diversity-only, ignore impact)
        # Familiar: alpha=1.0, gamma=0.3  (standard joint weights)
        if max_sim < low_threshold:
            regime = "novel"
            alpha = 2.0
            gamma = 0.0
        elif max_sim > high_threshold:
            regime = "familiar"
            alpha = 1.0
            gamma = 0.3
        else:
            regime = "interpolated"
            # Linear interpolation between thresholds
            t = (max_sim - low_threshold) / (high_threshold - low_threshold)
            alpha = 2.0 + t * (1.0 - 2.0)   # 2.0 -> 1.0
            gamma = 0.0 + t * (0.3 - 0.0)   # 0.0 -> 0.3

        # Save original weights, apply adaptive ones, select, restore
        orig_alpha, orig_gamma = self.alpha, self.gamma
        self.alpha = alpha
        self.gamma = gamma
        try:
            rec = self.select(mol_smiles, core_smiles, k=k, mmps=mmps)
        finally:
            self.alpha = orig_alpha
            self.gamma = orig_gamma

        return AdaptiveRecommendation(
            recommendation=rec,
            query_core=core_smiles,
            max_tanimoto=max_sim,
            regime=regime,
            alpha_used=alpha,
            gamma_used=gamma,
        )
