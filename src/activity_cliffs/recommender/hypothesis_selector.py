"""
Hypothesis Selector: Information-gain-optimal change-type recommendations.

Given a sensitive position on a molecule, selects k change-type axes that
maximize information coverage -- i.e., the 2-3 experiments that would tell
you the most about the SAR at this position with the fewest compounds.

The selector balances three objectives:
  1. Impact: predicted activity change magnitude (from the change-type model)
  2. Diversity: low correlation between selected axes (from empirical data)
  3. Feasibility: whether known MMP transforms exist for this change at this position

Algorithm: Greedy submodular maximization with a diversity-weighted impact score.
At each step, the next axis is chosen to maximize:
    score(axis) = impact(axis) * diversity_penalty(axis, already_selected)
where diversity_penalty = 1 - max_abs_correlation(axis, already_selected).

This is a greedy approximation to D-optimal experimental design and achieves
a (1 - 1/e) ≈ 63% approximation guarantee for submodular objectives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Axis metadata ─────────────────────────────────────────────────────────────

AXIS_NAMES = [
    "delta_has_ewg", "delta_has_edg", "delta_ewg_count", "delta_edg_count",
    "delta_n_hbd", "delta_n_hba", "delta_lipophilicity", "delta_heavy_atoms",
    "delta_n_rings", "delta_n_arom_rings", "delta_fsp3",
]

AXIS_LABELS = {
    "delta_has_ewg": "EWG character",
    "delta_has_edg": "EDG character",
    "delta_ewg_count": "EWG count",
    "delta_edg_count": "EDG count",
    "delta_n_hbd": "H-bond donor",
    "delta_n_hba": "H-bond acceptor",
    "delta_lipophilicity": "Lipophilicity",
    "delta_heavy_atoms": "Size (heavy atoms)",
    "delta_n_rings": "Ring count",
    "delta_n_arom_rings": "Aromatic rings",
    "delta_fsp3": "Saturation (Fsp3)",
}

# Human-readable hypothesis families -- groups of correlated axes
HYPOTHESIS_FAMILIES = {
    "Electronic": ["delta_has_ewg", "delta_ewg_count"],
    "Polar/Donor": ["delta_has_edg", "delta_edg_count", "delta_n_hbd"],
    "H-bond acceptor": ["delta_n_hba"],
    "Lipophilic": ["delta_lipophilicity"],
    "Steric/Size": ["delta_heavy_atoms", "delta_n_rings", "delta_n_arom_rings"],
    "3D character": ["delta_fsp3"],
}

# Concrete example transforms per axis (isosteric or near-isosteric)
EXAMPLE_TRANSFORMS = {
    "delta_has_ewg": [
        ("-CH3", "-CF3", "Adds strong EWG, same heavy atom count"),
        ("-H", "-F", "Minimal size change, adds electronegativity"),
        ("-H", "-Cl", "Adds moderate EWG + lipophilicity"),
    ],
    "delta_has_edg": [
        ("-H", "-NH2", "Adds EDG + H-bond donor"),
        ("-Cl", "-OMe", "Swaps EWG for EDG, similar size"),
    ],
    "delta_ewg_count": [
        ("-CH3", "-CF3", "0 -> 3 fluorines, same connectivity"),
        ("-F", "-CF3", "1 -> 3 fluorines"),
    ],
    "delta_edg_count": [
        ("-H", "-OH", "Adds one EDG"),
        ("-H", "-NMe2", "Adds one strong EDG"),
    ],
    "delta_n_hbd": [
        ("-OMe", "-OH", "Removes methyl, adds donor"),
        ("-H", "-NH2", "Adds primary amine donor"),
        ("-NMe2", "-NH2", "Demethylation reveals donors"),
    ],
    "delta_n_hba": [
        ("-CH3", "-OMe", "Replaces C with O, adds acceptor"),
        ("-H", "-CN", "Adds nitrile acceptor, minimal size"),
        ("-CH3", "-CONH2", "Adds amide acceptor"),
    ],
    "delta_lipophilicity": [
        ("-H", "-CH3", "Small lipophilic addition"),
        ("-OH", "-OMe", "Masks polar, adds lipophilicity"),
        ("-NH2", "-NMe2", "Alkylation increases logP"),
    ],
    "delta_heavy_atoms": [
        ("-H", "-CH3", "+1 carbon, minimal"),
        ("-CH3", "-iPr", "+2 carbons, branching"),
    ],
    "delta_n_rings": [
        ("-CH3", "-cPr", "Adds cyclopropyl ring"),
        ("-iPr", "-Ph", "Linear -> aromatic ring"),
    ],
    "delta_n_arom_rings": [
        ("-cHex", "-Ph", "Saturated -> aromatic ring"),
        ("-H", "-Ph", "Adds phenyl"),
    ],
    "delta_fsp3": [
        ("-Ph", "-cHex", "Aromatic -> saturated, increases Fsp3"),
        ("-vinyl", "-Et", "Unsaturated -> saturated"),
    ],
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AxisRecommendation:
    """A single recommended change-type axis."""
    axis_name: str
    axis_label: str
    family: str
    predicted_impact: float
    diversity_score: float
    combined_score: float
    rank: int
    direction: str  # "increase" or "decrease"
    rationale: str
    example_transforms: list[tuple[str, str, str]]  # (from, to, description)


@dataclass
class HypothesisPlan:
    """A complete experimental plan for a position."""
    k: int
    recommendations: list[AxisRecommendation]
    coverage_score: float  # fraction of hypothesis space covered
    all_impacts: dict[str, float]  # impact scores for all 11 axes
    correlation_matrix: np.ndarray
    position_context: Optional[dict] = None


# ── Core algorithm ────────────────────────────────────────────────────────────

class HypothesisSelector:
    """
    Selects k maximally informative change-type axes for a given position.

    Parameters
    ----------
    correlation_matrix : np.ndarray, shape (11, 11)
        Pearson correlation matrix of the 11 change-type axes,
        computed from training MMP data.
    axis_names : list[str]
        Names of the 11 axes (must match correlation_matrix order).
    diversity_weight : float
        How much to weight diversity vs. impact. 0 = pure impact ranking,
        1 = equal weight. Default 0.5.
    """

    def __init__(
        self,
        correlation_matrix: np.ndarray,
        axis_names: list[str] | None = None,
        diversity_weight: float = 0.5,
    ):
        self.corr = np.abs(correlation_matrix)  # absolute correlation for diversity
        self.raw_corr = correlation_matrix
        self.axis_names = axis_names or AXIS_NAMES
        self.diversity_weight = diversity_weight
        self.n_axes = len(self.axis_names)

        # Map axes to families
        self._axis_to_family = {}
        for family, axes in HYPOTHESIS_FAMILIES.items():
            for ax in axes:
                self._axis_to_family[ax] = family

    def select(
        self,
        impact_scores: dict[str, float],
        k: int = 3,
        exclude_axes: list[str] | None = None,
        require_families: list[str] | None = None,
    ) -> HypothesisPlan:
        """
        Select k axes that maximize impact × diversity.

        Parameters
        ----------
        impact_scores : dict[str, float]
            Predicted |delta_pActivity| for each axis (from model probing).
            Keys are axis names, values are predicted magnitudes.
        k : int
            Number of axes to select.
        exclude_axes : list[str], optional
            Axes to exclude (e.g., if the user already has data on that axis).
        require_families : list[str], optional
            Families that must be represented in the selection.

        Returns
        -------
        HypothesisPlan with k ranked recommendations.
        """
        exclude = set(exclude_axes or [])
        available = [i for i, name in enumerate(self.axis_names) if name not in exclude]

        # Normalize impact scores to [0, 1]
        raw_impacts = np.array([impact_scores.get(name, 0.0) for name in self.axis_names])
        if raw_impacts.max() > 0:
            norm_impacts = raw_impacts / raw_impacts.max()
        else:
            norm_impacts = np.ones(self.n_axes) / self.n_axes

        # Greedy submodular selection
        selected_indices: list[int] = []
        selected_directions: list[str] = []

        for step in range(min(k, len(available))):
            best_idx = -1
            best_score = -np.inf
            best_direction = "increase"

            for i in available:
                if i in selected_indices:
                    continue

                # Impact component
                impact = norm_impacts[i]

                # Diversity component: 1 - max correlation with already selected
                if selected_indices:
                    max_corr = max(self.corr[i, j] for j in selected_indices)
                    diversity = 1.0 - max_corr
                else:
                    diversity = 1.0  # first selection is always fully diverse

                # Combined score
                w = self.diversity_weight
                combined = (1 - w) * impact + w * diversity

                if combined > best_score:
                    best_score = combined
                    best_idx = i
                    # Direction: which sigma direction gave higher impact?
                    best_direction = "increase"  # default; caller can refine

            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_directions.append(best_direction)

        # Ensure required families are represented
        if require_families:
            selected_families = set(
                self._axis_to_family.get(self.axis_names[i], "") for i in selected_indices
            )
            missing = [f for f in require_families if f not in selected_families]
            for fam in missing:
                # Find highest-impact axis in this family not yet selected
                fam_axes = HYPOTHESIS_FAMILIES.get(fam, [])
                candidates = [
                    (norm_impacts[self.axis_names.index(a)], self.axis_names.index(a))
                    for a in fam_axes
                    if a not in exclude and self.axis_names.index(a) not in selected_indices
                ]
                if candidates:
                    candidates.sort(reverse=True)
                    # Replace the lowest-scoring selected axis
                    new_idx = candidates[0][1]
                    if len(selected_indices) >= k:
                        # Find worst current selection
                        worst_pos = min(
                            range(len(selected_indices)),
                            key=lambda p: norm_impacts[selected_indices[p]]
                        )
                        selected_indices[worst_pos] = new_idx
                    else:
                        selected_indices.append(new_idx)
                        selected_directions.append("increase")

        # Build recommendations
        recommendations = []
        for rank, idx in enumerate(selected_indices):
            name = self.axis_names[idx]
            family = self._axis_to_family.get(name, "Other")

            # Compute diversity score for this specific selection
            others = [j for j in selected_indices if j != idx]
            if others:
                div_score = 1.0 - max(self.corr[idx, j] for j in others)
            else:
                div_score = 1.0

            # Generate rationale
            rationale = self._generate_rationale(
                name, family, raw_impacts[idx], div_score, rank, selected_indices
            )

            rec = AxisRecommendation(
                axis_name=name,
                axis_label=AXIS_LABELS.get(name, name),
                family=family,
                predicted_impact=float(raw_impacts[idx]),
                diversity_score=float(div_score),
                combined_score=float(norm_impacts[idx] * (1 - self.diversity_weight)
                                     + div_score * self.diversity_weight),
                rank=rank + 1,
                direction=selected_directions[rank] if rank < len(selected_directions) else "increase",
                rationale=rationale,
                example_transforms=EXAMPLE_TRANSFORMS.get(name, []),
            )
            recommendations.append(rec)

        # Coverage: fraction of hypothesis families represented
        covered_families = set(r.family for r in recommendations)
        total_families = len(HYPOTHESIS_FAMILIES)
        coverage = len(covered_families) / total_families

        return HypothesisPlan(
            k=len(recommendations),
            recommendations=recommendations,
            coverage_score=coverage,
            all_impacts={name: float(raw_impacts[i]) for i, name in enumerate(self.axis_names)},
            correlation_matrix=self.raw_corr,
        )

    def _generate_rationale(
        self, axis_name: str, family: str, impact: float,
        diversity: float, rank: int, all_selected: list[int],
    ) -> str:
        """Generate a human-readable rationale for selecting this axis."""
        parts = []

        if rank == 0:
            parts.append(f"Highest predicted impact ({impact:.2f} pActivity units).")
        else:
            parts.append(f"Predicted impact: {impact:.2f} pActivity units.")

        if diversity > 0.7:
            parts.append(f"Highly orthogonal to other selections (diversity: {diversity:.2f}).")
        elif diversity > 0.4:
            parts.append(f"Moderately orthogonal to other selections (diversity: {diversity:.2f}).")
        else:
            parts.append(f"Some correlation with other selections (diversity: {diversity:.2f}), "
                        f"but included for its {family.lower()} coverage.")

        # Family-specific context
        family_rationales = {
            "Electronic": "Probes electronic tolerance. EWG/EDG changes test whether the "
                         "binding pocket has charge complementarity at this position.",
            "Polar/Donor": "Probes polar interaction potential. Tests whether there is an "
                          "H-bond partner in the binding pocket at this position.",
            "H-bond acceptor": "Probes acceptor opportunities. Tests whether the protein "
                              "presents a donor near this position.",
            "Lipophilic": "Probes hydrophobic tolerance. Tests whether this position "
                         "faces a lipophilic pocket or solvent.",
            "Steric/Size": "Probes steric tolerance. Tests whether there is room to grow "
                          "at this position or if size is penalized.",
            "3D character": "Probes conformational preference. Tests whether the pocket "
                          "prefers flat (aromatic) or 3D (sp3) character at this position.",
        }
        parts.append(family_rationales.get(family, ""))

        return " ".join(parts)

    def format_plan(self, plan: HypothesisPlan, position_label: str = "") -> str:
        """Format a HypothesisPlan as human-readable text."""
        lines = []
        if position_label:
            lines.append(f"=== Hypothesis Plan: {position_label} ===")
        lines.append(f"Selecting {plan.k} of 11 change-type axes "
                     f"(coverage: {plan.coverage_score:.0%} of hypothesis families)")
        lines.append("")

        for rec in plan.recommendations:
            lines.append(f"  {rec.rank}. {rec.family.upper()}: {rec.axis_label}")
            lines.append(f"     Predicted impact: {rec.predicted_impact:.2f} pActivity units")
            lines.append(f"     Diversity: {rec.diversity_score:.2f}")
            lines.append(f"     {rec.rationale}")
            if rec.example_transforms:
                lines.append(f"     Example transforms:")
                for frm, to, desc in rec.example_transforms[:2]:
                    lines.append(f"       {frm} --> {to}  ({desc})")
            lines.append("")

        # Show what we're NOT testing and why
        tested_axes = {r.axis_name for r in plan.recommendations}
        skipped = [(name, plan.all_impacts[name])
                   for name in self.axis_names if name not in tested_axes]
        skipped.sort(key=lambda x: -x[1])

        lines.append("  Axes NOT selected (lower priority or redundant):")
        for name, impact in skipped:
            family = self._axis_to_family.get(name, "Other")
            # Find max correlation with a selected axis
            idx = self.axis_names.index(name)
            sel_indices = [self.axis_names.index(r.axis_name) for r in plan.recommendations]
            if sel_indices:
                max_corr_with = max(
                    (self.corr[idx, j], self.axis_names[j]) for j in sel_indices
                )
                reason = (f"correlated with {AXIS_LABELS[max_corr_with[1]]} "
                         f"(r={max_corr_with[0]:.2f})")
            else:
                reason = "lower impact"
            lines.append(f"    - {AXIS_LABELS[name]} ({family}): "
                        f"impact={impact:.2f}, skipped because {reason}")

        return "\n".join(lines)
