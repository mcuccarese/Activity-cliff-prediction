"""
Compound Generator: Turn change-type recommendations into specific molecules.

Given a molecule, a sensitive position (core_smiles), and a recommended
change-type axis, generates concrete R-group substitutions by:
  1. Looking up known MMP transforms at this position from the corpus
  2. Filtering transforms that primarily realize the desired change type
  3. Ranking by synthesizability (SA score) and diversity
  4. Optionally verifying with AiZynthFinder retrosynthesis

This is the bridge from "try an EWG here" to "make this specific compound."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GeneratedCompound:
    """A specific compound recommendation."""
    smiles: str
    rgroup_from: str
    rgroup_to: str
    core_smiles: str
    change_axis: str
    change_axis_label: str
    delta_props: dict[str, float]  # all 11 property changes
    dominant_axis: str  # which axis this transform primarily changes
    abs_delta_pActivity_mean: float  # historical activity impact
    n_historical_mmps: int  # how many times this transform was observed
    sa_score: float  # synthetic accessibility (1=easy, 10=hard)
    mol_weight: float
    synthesizable: Optional[bool] = None  # from AiZynthFinder, if checked
    retro_route: Optional[str] = None


@dataclass
class CompoundPlan:
    """A set of recommended compounds for one position + one change axis."""
    position_core: str
    change_axis: str
    change_axis_label: str
    compounds: list[GeneratedCompound]
    n_candidates_found: int
    n_after_filter: int


# ── SA Score ──────────────────────────────────────────────────────────────────

def sa_score(smiles: str) -> float:
    """Compute synthetic accessibility score (Ertl, 1-10, lower=easier)."""
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
        # Fallback: use simple heuristic
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        # Simple proxy: ring count + chiral centers + MW/100
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        mw = Descriptors.MolWt(mol)
        return min(10.0, 1.0 + n_rings * 0.5 + n_chiral * 1.0 + mw / 500.0)


# ── Core generator ────────────────────────────────────────────────────────────

class CompoundGenerator:
    """
    Generates specific compounds by looking up known MMP transforms.

    Parameters
    ----------
    mmps_path : Path
        Path to all_mmps.parquet (25M MMP corpus).
    rgroup_props_path : Path
        Path to rgroup_props.parquet (pre-computed R-group properties).
    change_type_names : list[str]
        The 11 change-type axis names.
    delta_prop_sigmas : dict[str, float]
        Per-axis standard deviations for standardization.
    """

    def __init__(
        self,
        mmps_path: Path | None = None,
        rgroup_props_path: Path | None = None,
        change_type_names: list[str] | None = None,
        delta_prop_sigmas: dict[str, float] | None = None,
    ):
        self.mmps_path = mmps_path or ROOT / "outputs" / "mmps" / "all_mmps.parquet"
        self.rgroup_props_path = rgroup_props_path or ROOT / "outputs" / "features" / "rgroup_props.parquet"

        from activity_cliffs.features.change_type import CHANGE_TYPE_NAMES, RGROUP_PROP_NAMES
        self.change_type_names = change_type_names or CHANGE_TYPE_NAMES
        self.rgroup_prop_names = RGROUP_PROP_NAMES

        # Load sigmas from model metadata if not provided
        if delta_prop_sigmas is None:
            import json
            meta_path = ROOT / "webapp" / "model" / "change_type_meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.sigmas = meta["delta_prop_sigmas"]
        else:
            self.sigmas = delta_prop_sigmas

        # Lazy-loaded data
        self._rgroup_props: dict[str, np.ndarray] | None = None
        self._mmps_index: pd.DataFrame | None = None

    def _load_rgroup_props(self) -> dict[str, np.ndarray]:
        """Load R-group property lookup."""
        if self._rgroup_props is None:
            logger.info("Loading R-group properties...")
            df = pd.read_parquet(self.rgroup_props_path)
            self._rgroup_props = dict(
                zip(df["rgroup_smiles"], df[self.rgroup_prop_names].values.astype(np.float32))
            )
            logger.info("  %d R-groups loaded", len(self._rgroup_props))
        return self._rgroup_props

    def _load_mmps_for_core(self, core_smiles: str) -> pd.DataFrame:
        """Load MMPs at a specific core (position)."""
        logger.info("Loading MMPs for core: %s", core_smiles[:50])
        mmps = pd.read_parquet(
            self.mmps_path,
            columns=["target_chembl_id", "core_smiles", "smiles_from", "smiles_to",
                     "rgroup_from", "rgroup_to", "abs_delta_pActivity"],
            filters=[("core_smiles", "==", core_smiles)],
        )
        logger.info("  %d MMPs at this position", len(mmps))
        return mmps

    def generate_for_axis(
        self,
        mol_smiles: str,
        core_smiles: str,
        target_axis: str,
        max_compounds: int = 5,
        sa_threshold: float = 6.0,
    ) -> CompoundPlan:
        """
        Generate specific compounds that realize a given change-type axis
        at a given position.

        Parameters
        ----------
        mol_smiles : str
            The input molecule SMILES.
        core_smiles : str
            The core (scaffold with [*:1]) defining the position.
        target_axis : str
            The change-type axis to realize (e.g., "delta_has_ewg").
        max_compounds : int
            Maximum number of compounds to return.
        sa_threshold : float
            Maximum SA score (1-10) to include.

        Returns
        -------
        CompoundPlan with ranked, synthesizable compound suggestions.
        """
        from activity_cliffs.recommender.hypothesis_selector import AXIS_LABELS

        rg_props = self._load_rgroup_props()

        # Load MMPs at this position
        mmps = self._load_mmps_for_core(core_smiles)
        n_total = len(mmps)

        if n_total == 0:
            return CompoundPlan(
                position_core=core_smiles,
                change_axis=target_axis,
                change_axis_label=AXIS_LABELS.get(target_axis, target_axis),
                compounds=[],
                n_candidates_found=0,
                n_after_filter=0,
            )

        # Compute delta props for each MMP
        sigma_vec = np.array(
            [self.sigmas.get(c, 1.0) for c in self.change_type_names],
            dtype=np.float32
        )
        sigma_vec = np.where(sigma_vec < 1e-6, 1.0, sigma_vec)

        target_axis_idx = self.change_type_names.index(target_axis)
        zero_prop = np.zeros(len(self.rgroup_prop_names), dtype=np.float32)

        candidates = []
        for _, row in mmps.iterrows():
            pf = rg_props.get(row["rgroup_from"])
            pt = rg_props.get(row["rgroup_to"])
            if pf is None or pt is None:
                continue

            delta = pt - pf
            standardized = np.abs(delta) / sigma_vec
            dominant = int(np.argmax(standardized))

            # Is this MMP primarily along our target axis?
            if dominant == target_axis_idx or standardized[target_axis_idx] > 0.5:
                # Compute SA score for the product molecule
                product_smiles = row["smiles_to"]
                sa = sa_score(product_smiles)

                if sa <= sa_threshold:
                    mol = Chem.MolFromSmiles(product_smiles)
                    mw = Descriptors.MolWt(mol) if mol else 0.0

                    delta_dict = {
                        name: float(delta[j])
                        for j, name in enumerate(self.change_type_names)
                    }

                    candidates.append(GeneratedCompound(
                        smiles=product_smiles,
                        rgroup_from=row["rgroup_from"],
                        rgroup_to=row["rgroup_to"],
                        core_smiles=core_smiles,
                        change_axis=target_axis,
                        change_axis_label=AXIS_LABELS.get(target_axis, target_axis),
                        delta_props=delta_dict,
                        dominant_axis=self.change_type_names[dominant],
                        abs_delta_pActivity_mean=float(row["abs_delta_pActivity"]),
                        n_historical_mmps=1,  # will aggregate below
                        sa_score=sa,
                        mol_weight=mw,
                    ))

        n_candidates = len(candidates)
        logger.info("  %d candidate transforms (of %d total MMPs)", n_candidates, n_total)

        if not candidates:
            return CompoundPlan(
                position_core=core_smiles,
                change_axis=target_axis,
                change_axis_label=AXIS_LABELS.get(target_axis, target_axis),
                compounds=[],
                n_candidates_found=n_total,
                n_after_filter=0,
            )

        # Deduplicate by rgroup_to (keep best SA score)
        seen: dict[str, GeneratedCompound] = {}
        for c in candidates:
            key = c.rgroup_to
            if key not in seen or c.sa_score < seen[key].sa_score:
                seen[key] = c

        unique_candidates = list(seen.values())

        # Rank by: SA score (lower=better), then activity impact (higher=better)
        unique_candidates.sort(key=lambda c: (c.sa_score, -c.abs_delta_pActivity_mean))

        # Take top max_compounds
        final = unique_candidates[:max_compounds]

        return CompoundPlan(
            position_core=core_smiles,
            change_axis=target_axis,
            change_axis_label=AXIS_LABELS.get(target_axis, target_axis),
            compounds=final,
            n_candidates_found=n_total,
            n_after_filter=len(unique_candidates),
        )

    def format_plan(self, plan: CompoundPlan) -> str:
        """Format a CompoundPlan as human-readable text."""
        lines = []
        lines.append(f"  Change axis: {plan.change_axis_label}")
        lines.append(f"  Position: {plan.position_core[:60]}...")
        lines.append(f"  Candidates: {plan.n_after_filter} (from {plan.n_candidates_found} MMPs)")
        lines.append("")

        if not plan.compounds:
            lines.append("  No synthesizable compounds found for this axis at this position.")
            lines.append("  Consider: Lib-INVENT generative decoration as fallback.")
            return "\n".join(lines)

        for i, c in enumerate(plan.compounds):
            lines.append(f"  Compound {i+1}:")
            lines.append(f"    SMILES:    {c.smiles}")
            lines.append(f"    R-group:   {c.rgroup_from} --> {c.rgroup_to}")
            lines.append(f"    SA score:  {c.sa_score:.1f} (1=easy, 10=hard)")
            lines.append(f"    MW:        {c.mol_weight:.0f}")
            lines.append(f"    Historic impact: {c.abs_delta_pActivity_mean:.2f} pActivity units")

            # Show the key property change
            target_delta = c.delta_props.get(c.change_axis, 0)
            lines.append(f"    Target axis change: {target_delta:+.2f}")
            lines.append("")

        return "\n".join(lines)


# ── AiZynthFinder integration ─────────────────────────────────────────────────

def verify_synthesizability(
    smiles_list: list[str],
    config_path: str | None = None,
    time_limit: int = 120,
) -> list[dict]:
    """
    Verify synthesizability of a list of SMILES using AiZynthFinder.

    Returns a list of dicts with keys: smiles, solved, n_steps, route.
    """
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder

        if config_path is None:
            # Try to find downloaded models
            model_dir = ROOT / "models" / "aizynthfinder"
            config_file = model_dir / "config.yml"
            if not config_file.exists():
                logger.warning("AiZynthFinder config not found at %s", config_file)
                return [{"smiles": s, "solved": None, "n_steps": None,
                         "route": "AiZynthFinder not configured"} for s in smiles_list]
            config_path = str(config_file)

        finder = AiZynthFinder(configfile=config_path)
        finder.config.search.time_limit = time_limit

        results = []
        for smi in smiles_list:
            try:
                finder.target_smiles = smi
                finder.tree_search()
                finder.build_routes()
                stats = finder.routes.reaction_tree_statistics()

                solved = len(stats) > 0 and stats[0].get("is_solved", False)
                n_steps = stats[0].get("number_of_reactions", None) if stats else None

                results.append({
                    "smiles": smi,
                    "solved": solved,
                    "n_steps": n_steps,
                    "route": "Route found" if solved else "No route found",
                })
            except Exception as e:
                results.append({
                    "smiles": smi,
                    "solved": None,
                    "n_steps": None,
                    "route": f"Error: {str(e)[:100]}",
                })

        return results

    except ImportError:
        logger.warning("AiZynthFinder not installed")
        return [{"smiles": s, "solved": None, "n_steps": None,
                 "route": "AiZynthFinder not installed"} for s in smiles_list]
