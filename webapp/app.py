"""
SAR Sensitivity Explorer — M8 Interactive Explainability

Input a SMILES → see which positions on the molecule are most sensitive
to structural modification, based on 25M matched molecular pairs from
50 ChEMBL targets.

Run:
    streamlit run webapp/app.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# Ensure the project source is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.predict import (
    predict_positions,
    sensitivity_to_label,
    PositionResult,
    EvidenceExample,
    ChangeTypeRec,
    FEATURE_NAMES,
    get_explainer,
    CHANGE_TYPE_META_PATH,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SAR Cliff Explorer",
    page_icon="🧪",
    layout="wide",
)

# ── Load model metadata ─────────────────────────────────────────────────────

META_PATH = Path(__file__).parent / "model" / "model_meta.json"
with open(META_PATH) as f:
    MODEL_META = json.load(f)

# ── Example molecules ────────────────────────────────────────────────────────

EXAMPLES = {
    # Approved drugs
    "Imatinib (BCR-ABL)": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Celecoxib (COX-2)": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "Sorafenib (multi-kinase)": "CNC(=O)C1=CC(=CC=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
    # Screening deck scaffolds
    "Benzimidazole sulfonamide": "Cc1nc2ccccc2n1S(=O)(=O)Nc1ccc(F)cc1",
    "Piperazine aryl amide": "O=C(c1ccc(F)cc1)N1CCN(c2ccc(Cl)cc2)CC1",
    "Pyrimidine aniline": "Cc1ccnc(Nc2ccc(Cl)cc2F)n1",
    "Morpholine oxadiazole": "Cc1nc(-c2ccc(N3CCOCC3)cc2)no1",
    "Thiophene carboxamide": "O=C(Nc1cccc(Cl)c1)c1cccs1",
    "Triazole phenyl ether": "Clc1ccc(Oc2cccc(-n3cncn3)c2)cc1",
}

# ── Feature descriptions for chemists ────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "ctx_n_donor_4A": ("H-bond donors within 4\u00c5", "Nearby NH/OH groups that can donate H-bonds"),
    "ctx_n_acceptor_4A": ("H-bond acceptors within 4\u00c5", "Nearby N/O/halogen atoms with lone pairs"),
    "ctx_n_hydrophobic_4A": ("Hydrophobic atoms within 4\u00c5", "Nearby aliphatic carbons (no heteroatom neighbors)"),
    "ctx_n_aromatic_4A": ("Aromatic atoms within 4\u00c5", "Nearby aromatic ring atoms"),
    "ctx_sasa_attach": ("Solvent-accessible surface area", "How exposed the attachment point is (\u00c5\u00b2)"),
    "ctx_gasteiger_charge": ("Gasteiger partial charge", "Electronic character at attachment (- = electron-rich, + = electron-poor)"),
    "ctx_n_rotbonds_2": ("Rotatable bonds nearby", "Flexibility within 2 bonds of attachment"),
    "ctx_is_aromatic_attach": ("Aromatic attachment", "Whether the attachment atom is in an aromatic ring"),
    "ctx_n_heavy_4A": ("Heavy atoms within 4\u00c5", "Steric crowding around the attachment point"),
    "core_n_heavy": ("Core heavy atom count", "Size of the scaffold (larger = less sensitive)"),
    "core_n_rings": ("Core ring count", "Ring complexity of the scaffold"),
}


# ── Visualization ────────────────────────────────────────────────────────────

def sensitivity_color(value: float, vmin: float, vmax: float) -> tuple[float, float, float]:
    """Map sensitivity to a blue→white→red color. Returns (r, g, b) in [0, 1]."""
    if vmax == vmin:
        return (0.85, 0.85, 0.85)
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t * 2
        r = 0.2 + 0.8 * s
        g = 0.3 + 0.7 * s
        b = 0.9 + 0.1 * s
    else:
        s = (t - 0.5) * 2
        r = 1.0
        g = 1.0 - 0.7 * s
        b = 1.0 - 0.8 * s
    return (r, g, b)


def draw_molecule_with_sensitivity(
    smiles: str,
    results: list[PositionResult],
    width: int = 700,
    height: int = 420,
    show_rank_labels: bool = True,
    selected_rank: int | None = None,
) -> str:
    """Draw molecule SVG with atoms colored by predicted sensitivity.

    selected_rank: 0-based index into results for the currently selected position.
    That position gets a larger highlight radius to visually emphasize it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    atom_sens = {}
    atom_rank = {}
    for i, r in enumerate(results):
        atom_sens[r.atom_idx] = r.sensitivity
        atom_rank[r.atom_idx] = i + 1

    if not atom_sens:
        return ""

    vmin = min(atom_sens.values())
    vmax = max(atom_sens.values())

    atom_colors = {}
    atom_radii = {}
    highlight_atoms = []

    for i, r in enumerate(results):
        color = sensitivity_color(r.sensitivity, vmin, vmax)
        is_selected = (selected_rank is not None and i == selected_rank)
        atom_colors[r.atom_idx] = color
        atom_radii[r.atom_idx] = 0.65 if is_selected else 0.4
        highlight_atoms.append(r.atom_idx)

        if r.neighbor_idx not in atom_colors:
            nc = sensitivity_color(r.sensitivity, vmin, vmax)
            faded = tuple(0.5 + 0.5 * c for c in nc)
            atom_colors[r.neighbor_idx] = faded
            atom_radii[r.neighbor_idx] = 0.35 if is_selected else 0.25
            highlight_atoms.append(r.neighbor_idx)

    highlight_bonds = []
    bond_colors = {}
    for r in results:
        bond = mol.GetBondBetweenAtoms(r.atom_idx, r.neighbor_idx)
        if bond is not None:
            bidx = bond.GetIdx()
            highlight_bonds.append(bidx)
            bond_colors[bidx] = sensitivity_color(r.sensitivity, vmin, vmax)

    if show_rank_labels:
        for atom_idx, rank in atom_rank.items():
            mol.GetAtomWithIdx(atom_idx).SetProp("atomNote", f"#{rank}")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.15
    opts.additionalAtomLabelPadding = 0.1
    opts.annotationFontScale = 0.6

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _smiles_to_svg(smiles: str, width: int = 200, height: int = 140) -> str:
    """Render a SMILES string as an SVG. Returns empty string on failure."""
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().padding = 0.15
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def _svg_to_img_html(svg: str, width: str = "180px") -> str:
    """Convert an SVG string to a base64-encoded HTML img tag."""
    if not svg:
        return (
            f'<div style="width:{width};height:110px;background:#f3f4f6;'
            f'border-radius:4px;display:flex;align-items:center;'
            f'justify-content:center;color:#9ca3af;font-size:0.8em;">No structure</div>'
        )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" '
        f'style="background:white;border-radius:6px;border:1px solid #e5e7eb;" />'
    )


# ── ChEMBL link helpers ───────────────────────────────────────────────────────

def _target_link_html(target_id: str, target_name: str) -> str:
    """Return an HTML anchor linking to the ChEMBL target report card."""
    url = f"https://www.ebi.ac.uk/chembl/target_report_card/{target_id}"
    return (
        f'<a href="{url}" target="_blank" rel="noopener" '
        f'style="color:#1d4ed8;text-decoration:none;font-weight:bold;">'
        f'{target_name} ({target_id}) ↗</a>'
    )


def _compound_link_html(chembl_id: str) -> str:
    """Return an HTML anchor linking to the ChEMBL compound report card."""
    if not chembl_id:
        return ""
    url = f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}"
    return (
        f'<a href="{url}" target="_blank" rel="noopener" '
        f'style="color:#6b7280;font-size:0.78em;text-decoration:none;">'
        f'{chembl_id} ↗</a>'
    )


# ── Evidence rendering ────────────────────────────────────────────────────────

def _delta_color(delta: float) -> str:
    """CSS color for a ΔpActivity value."""
    if abs(delta) >= 1.5:
        return "#e63333" if delta > 0 else "#334de6"
    elif abs(delta) >= 1.0:
        return "#d97706" if delta > 0 else "#2563eb"
    else:
        return "#6b7280"


def _render_evidence(evidence: list[EvidenceExample]) -> None:
    """Render evidence examples as styled cards with structures and ChEMBL links."""
    exact = [e for e in evidence if e.source == "exact"]
    similar = [e for e in evidence if e.source == "similar"]

    def _render_group(examples: list[EvidenceExample], caption: str) -> None:
        if not examples:
            return
        st.caption(caption)
        for e in examples:
            delta_sign = "+" if e.delta_pActivity > 0 else ""
            color = _delta_color(e.delta_pActivity)

            # Prefer full molecule SMILES; fall back to R-group fragments
            use_full = bool(e.smiles_from and e.smiles_to)
            if use_full:
                svg_from = _smiles_to_svg(e.smiles_from, 200, 145)
                svg_to = _smiles_to_svg(e.smiles_to, 200, 145)
                img_width = "190px"
            else:
                svg_from = _smiles_to_svg(e.rgroup_from.replace("[*:1]", "[*]"), 130, 100)
                svg_to = _smiles_to_svg(e.rgroup_to.replace("[*:1]", "[*]"), 130, 100)
                img_width = "120px"

            img_from = _svg_to_img_html(svg_from, img_width)
            img_to = _svg_to_img_html(svg_to, img_width)

            link_from = _compound_link_html(e.molecule_chembl_id_from)
            link_to = _compound_link_html(e.molecule_chembl_id_to)

            sim_html = ""
            if e.source == "similar":
                sim_pct = e.similarity * 100
                sim_html = (
                    f'<span style="color:#9ca3af;font-size:0.82em;">'
                    f'({sim_pct:.0f}% pharmacophore similarity)</span>'
                )

            target_html = _target_link_html(e.target_id, e.target_name)

            card = (
                f'<div style="margin-bottom:10px;padding:10px 14px;background:#f8f9fa;'
                f'border-left:3px solid {color};border-radius:6px;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:8px;">'
                f'<div>{target_html} {sim_html}</div>'
                f'<div style="color:{color};font-weight:bold;font-size:1.05em;">'
                f'ΔpActivity = {delta_sign}{e.delta_pActivity:.2f}</div>'
                f'</div>'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<div style="text-align:center;">{img_from}'
                f'<div style="margin-top:3px;">{link_from}</div></div>'
                f'<div style="font-size:1.5em;color:#9ca3af;padding-bottom:20px;">→</div>'
                f'<div style="text-align:center;">{img_to}'
                f'<div style="margin-top:3px;">{link_to}</div></div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card, unsafe_allow_html=True)

    _render_group(exact, "Exact core match — same scaffold exists in ChEMBL:")
    _render_group(
        similar,
        "Similar pharmacophore context:" if not exact else "Similar positions on other scaffolds:",
    )
    if not exact and not similar:
        st.caption("No evidence examples found for this position.")


# ── SHAP attribution panel ────────────────────────────────────────────────────

def _render_attribution(result: PositionResult) -> None:
    """Render per-feature SHAP contributions as a horizontal bar chart.

    Red bars = feature raises predicted sensitivity.
    Blue bars = feature lowers predicted sensitivity.
    Bar width is proportional to the absolute SHAP value.
    """
    if not result.attribution:
        # Fall back to plain feature table when SHAP is unavailable
        feat_rows = []
        for fname in FEATURE_NAMES:
            val = result.features.get(fname, 0.0)
            short, desc = FEATURE_DESCRIPTIONS.get(fname, (fname, ""))
            feat_rows.append({"Feature": short, "Value": f"{val:.3f}", "Description": desc})
        st.dataframe(pd.DataFrame(feat_rows), hide_index=True, use_container_width=True)
        return

    items = sorted(result.attribution.items(), key=lambda x: abs(x[1]), reverse=True)
    max_abs = max(abs(v) for _, v in items) if items else 1.0
    delta = result.sensitivity - result.base_value
    sign = "+" if delta >= 0 else ""

    header = (
        f'<div style="font-size:0.82em;color:#6b7280;margin-bottom:8px;">'
        f"Model baseline: <b>{result.base_value:.2f}</b> &nbsp;→&nbsp; "
        f"This position: <b>{result.sensitivity:.2f}</b> "
        f'<span style="color:{"#b91c1c" if delta >= 0 else "#1d4ed8"};">'
        f"({sign}{delta:.2f})</span></div>"
    )

    rows = [header]
    for fname, shap_val in items:
        if abs(shap_val) < 0.005:
            continue
        short_name, desc = FEATURE_DESCRIPTIONS.get(fname, (fname, ""))
        feat_val = result.features.get(fname, 0.0)
        bar_pct = abs(shap_val) / max_abs * 100

        if shap_val > 0:
            bar_color = "#fca5a5"   # light red fill
            text_color = "#b91c1c"  # dark red label
            arrow = "▲"
        else:
            bar_color = "#93c5fd"   # light blue fill
            text_color = "#1d4ed8"  # dark blue label
            arrow = "▼"

        contribution_str = f"{arrow} {'+' if shap_val >= 0 else ''}{shap_val:.3f}"

        rows.append(
            f'<div style="margin-bottom:6px;" title="{desc} — value: {feat_val:.3f}">'
            f'  <div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'    <span style="font-size:0.82em;color:#374151;">{short_name}</span>'
            f'    <span style="font-size:0.82em;font-weight:600;color:{text_color};'
            f'min-width:60px;text-align:right;">{contribution_str}</span>'
            f"  </div>"
            f'  <div style="background:#f3f4f6;border-radius:3px;height:8px;margin-top:2px;">'
            f'    <div style="background:{bar_color};width:{bar_pct:.0f}%;height:100%;'
            f'border-radius:3px;"></div>'
            f"  </div>"
            f"</div>"
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


# ── Change type recommendation panel ────────────────────────────────────────

def _render_change_type_recs(recs: list[ChangeTypeRec], n_show: int = 8) -> None:
    """Render ranked change type recommendations as a cliff-magnitude bar chart.

    Bar width and color intensity represent predicted |ΔpActivity| at ±1σ of that
    property axis — i.e., how much activity tends to swing when that property is
    changed.  No direction is asserted: this is a 'start here' signal only.
    """
    if not recs:
        st.caption("Change type model not yet available — run train_change_type_model.py.")
        return

    top = recs[:n_show]
    max_score = max(r.cliff_score for r in top) if top else 1.0
    if max_score < 1e-6:
        max_score = 1.0

    rows: list[str] = []
    for r in top:
        pct = r.cliff_score / max_score * 100
        # Orange intensity scaled by cliff score (white → #f97316 orange)
        intensity = r.cliff_score / max_score
        r_ch = int(255)
        g_ch = int(255 - intensity * (255 - 115))   # 255 → 115
        b_ch = int(255 - intensity * 255)            # 255 → 0
        bar_color = f"rgb({r_ch},{g_ch},{b_ch})"
        text_color = "#9a3412" if intensity > 0.5 else "#374151"

        rows.append(
            f'<div style="margin-bottom:6px;">'
            f'  <div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'    <span style="font-size:0.82em;color:#374151;">{r.rank}. {r.label}</span>'
            f'    <span style="font-size:0.82em;font-weight:600;color:{text_color};">'
            f'|Δ| ≈ {r.cliff_score:.2f}</span>'
            f"  </div>"
            f'  <div style="background:#f3f4f6;border-radius:3px;height:8px;margin-top:2px;">'
            f'    <div style="background:{bar_color};width:{pct:.0f}%;height:100%;'
            f'border-radius:3px;"></div>'
            f"  </div>"
            f"</div>"
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


# ── Position pill badge ───────────────────────────────────────────────────────

def _badge(label: str) -> str:
    if label in ("Very High", "High"):
        return "🔴"
    elif label == "Moderate":
        return "🟡"
    else:
        return "🔵"


# ── Main app ─────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.title("SAR Cliff Explorer")
    st.markdown(
        "Predict which positions on a molecule are most prone to **activity cliffs** "
        "-- where small structural changes cause disproportionately large potency "
        "shifts. Based on **25 million matched molecular pairs** across 50 ChEMBL "
        "targets, normalized by modification size (SALI)."
    )

    with st.expander("About this tool"):
        st.markdown("""
**How it works**

This tool uses **SALI normalization** (Structure-Activity Landscape Index) to detect
genuine activity cliffs. Raw sensitivity (mean |delpActivity|) is confounded by
R-group size -- bigger structural changes trivially produce bigger potency shifts.
SALI divides by modification size, isolating positions where *small* changes cause
*disproportionately large* activity swings.

The ML model uses 3D pharmacophore context (Gasteiger charge, solvent exposure,
aromaticity, steric crowding) plus core topology to predict SALI sensitivity.
Under SALI normalization, the scaffold-size heuristic performs **worse than random**
(NDCG@3 = 0.791 vs random 0.839), while the ML model achieves **0.910** --
meaning the 3D features are the genuine predictive signal, not scaffold size.

This advantage holds out-of-distribution:

| Scenario | Random | Heuristic | ML model |
|:---|:---:|:---:|:---:|
| LOO-target (50 targets) | 0.839 | 0.791 | **0.910** |
| Novel scaffolds (20% held-out) | 0.872 | 0.839 | **0.913** |
| Temporal split (train ≤2015, 83% novel cores) | 0.806 | 0.755 | **0.878** |

The ML model wins in every scenario, and the advantage **grows** further
from the training distribution (+7% in-distribution, +12% on future chemistry).

**What this tool IS:**
- A cliff detector: finds positions where small changes move potency the most
- Backed by 25M real matched molecular pairs from ChEMBL
- Target-agnostic: works for any target without retraining
- Validated on novel scaffolds and future chemical matter

**What this tool is NOT:**
- Not a potency predictor (predicts cliff magnitude, not direction)
- Not a substitute for protein-aware modeling
- Not validated prospectively in a drug discovery campaign
        """)


    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("About the Model")
        st.markdown(f"""
**Training data:** {MODEL_META['n_training_rows']:,} position observations
across {MODEL_META['n_targets_trained_on']} protein targets

**Validation (leave-one-target-out):**
- NDCG@3 = {MODEL_META['validation_metrics']['ndcg_at_3']:.4f}
- Hit@1 = {MODEL_META['validation_metrics']['hit_at_1']:.0%}
- Spearman = {MODEL_META['validation_metrics']['spearman']:.3f}

**What it predicts:** SALI sensitivity \u2014 the expected potency
change *per unit of structural modification*. Higher values =
small R-group changes cause disproportionately large potency
shifts (activity cliffs).

**Key finding:** Under SALI normalization, scaffold size alone
performs **worse than random**. The genuine signal comes from
3D pharmacophore context \u2014 ring complexity, electronic character,
and steric environment at the attachment point.
        """)

        st.markdown("**Validation (NDCG@3)**")
        _val_table = (
            '<table style="width:100%;font-size:0.82em;border-collapse:collapse;">'
            '<tr style="border-bottom:2px solid #e5e7eb;">'
            '<th style="text-align:left;padding:4px 6px;">Scenario</th>'
            '<th style="text-align:center;padding:4px 4px;">Random</th>'
            '<th style="text-align:center;padding:4px 4px;">Heuristic</th>'
            '<th style="text-align:center;padding:4px 4px;">ML</th>'
            '</tr>'
            '<tr style="border-bottom:1px solid #f3f4f6;">'
            '<td style="padding:4px 6px;">LOO-target</td>'
            '<td style="text-align:center;padding:4px;">0.839</td>'
            '<td style="text-align:center;padding:4px;color:#dc2626;">0.791</td>'
            '<td style="text-align:center;padding:4px;color:#16a34a;font-weight:bold;">0.910</td>'
            '</tr>'
            '<tr style="border-bottom:1px solid #f3f4f6;">'
            '<td style="padding:4px 6px;">Novel scaffolds</td>'
            '<td style="text-align:center;padding:4px;">0.872</td>'
            '<td style="text-align:center;padding:4px;color:#dc2626;">0.839</td>'
            '<td style="text-align:center;padding:4px;color:#16a34a;font-weight:bold;">0.913</td>'
            '</tr>'
            '<tr>'
            '<td style="padding:4px 6px;">Temporal split</td>'
            '<td style="text-align:center;padding:4px;">0.806</td>'
            '<td style="text-align:center;padding:4px;color:#dc2626;">0.755</td>'
            '<td style="text-align:center;padding:4px;color:#16a34a;font-weight:bold;">0.878</td>'
            '</tr>'
            '</table>'
            '<div style="font-size:0.75em;color:#9ca3af;margin-top:4px;">'
            'Heuristic = -core_n_heavy (scaffold size). '
            'Red = worse than random. Green = best.'
            '</div>'
        )
        st.markdown(_val_table, unsafe_allow_html=True)

        st.divider()
        st.subheader("Real-World Meaning")
        st.markdown("""
**Why the scores look high:** Most molecules have only 3\u20135 modifiable
positions. Ranking 4 items, even a random shuffle scores ~0.84 NDCG\u200a\u2014\u200athere
just aren't many ways to be wrong. The meaningful signal is the
**lift above random**.

**In practical terms (Hit@1):**
- Random guess: picks the cliff-prone position first **25%** of the time
- This model: picks it first **48%** of the time (**1.9\u00d7 better**)

**Compound savings:**
For a typical molecule with 4 positions, you'd need to explore
~2.5 positions randomly to find where the cliffs are.
The model cuts this to ~1.9 \u2014 about **1 fewer analog series per scaffold**.

Across a 10-scaffold campaign at ~10 compounds per position,
that's roughly **60 fewer compounds** to learn the same thing
\u2014 or equivalently, you reach the steep SAR **25% faster**.
        """)

        st.divider()
        st.subheader("Top Predictive Features")
        if "feature_importances" in MODEL_META:
            imps = MODEL_META["feature_importances"]
            for name, val in sorted(imps.items(), key=lambda x: -x[1])[:5]:
                short_name = FEATURE_DESCRIPTIONS.get(name, (name, ""))[0]
                st.markdown(f"- **{short_name}**: {val:.1%}")

        if CHANGE_TYPE_META_PATH.exists():
            st.divider()
            with open(CHANGE_TYPE_META_PATH) as _f:
                _ct_meta = json.load(_f)
            loo_mean = _ct_meta.get("loo_spearman_mean")
            loo_std  = _ct_meta.get("loo_spearman_std")
            n_rows   = _ct_meta.get("n_training_rows", 0)
            st.markdown("**Change type model (M9):**")
            st.markdown(f"Trained on {n_rows:,} MMPs")
            if loo_mean is not None:
                st.markdown(f"LOO-target Spearman = {loo_mean:.3f} ± {loo_std:.3f}")

        st.divider()
        st.markdown(
            "_SALI sensitivity = mean(|ΔpActivity| / (max R-group size + 1)) "
            "across all known modifications at equivalent positions in ChEMBL. "
            "Higher values indicate positions where small structural changes "
            "cause disproportionately large potency shifts — the hallmark of "
            "an activity cliff._"
        )

    # ── Input ─────────────────────────────────────────────────────────────
    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Example molecules**")
        for name, smi in EXAMPLES.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                st.session_state["smiles_input"] = smi

    with col_input:
        smiles = st.text_input(
            "Enter SMILES",
            value=st.session_state.get("smiles_input", ""),
            placeholder="e.g. CC1=C(C=C(C=C1)NC(=O)...",
            help="Paste a SMILES string for any drug-like molecule",
        )

    if not smiles:
        st.info("Enter a SMILES or click an example molecule to get started.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES. Please check the input.")
        return

    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rotatable > 5:
        st.warning(
            f"This molecule has **{n_rotatable} rotatable bonds**. 3D features are "
            "computed from a single conformer and may be less reliable for flexible "
            "molecules. Position ranking by core size remains valid."
        )

    # ── Predict ───────────────────────────────────────────────────────────
    with st.spinner("Generating conformer and computing features..."):
        results = predict_positions(smiles)

    if not results:
        st.warning(
            "No fragmentable positions found. The molecule may be too small "
            "or have no acyclic single bonds between heavy atoms."
        )
        return

    # ── Build pill options ────────────────────────────────────────────────
    pill_options = [
        f"#{i + 1} {_badge(sensitivity_to_label(r.sensitivity))} {r.sensitivity:.3f}"
        for i, r in enumerate(results)
    ]

    # Reset pill selection when SMILES changes
    if st.session_state.get("_pills_smiles") != smiles:
        st.session_state["_pills_smiles"] = smiles
        st.session_state["pos_pills"] = pill_options[0]

    # Read current selection before rendering pills (so SVG can use it)
    current_pill = st.session_state.get("pos_pills", pill_options[0])
    if current_pill not in pill_options:
        current_pill = pill_options[0]
    selected_rank = pill_options.index(current_pill)

    # ── Molecule SVG (centered, fixed size) ───────────────────────────────
    st.subheader("Cliff Sensitivity Map")
    _, col_mol, _ = st.columns([1, 3, 1])
    with col_mol:
        svg = draw_molecule_with_sensitivity(
            smiles, results, width=480, height=300, selected_rank=selected_rank
        )
        if svg:
            st.image(svg, use_container_width=True)
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-top:-6px;">'
            '<span style="color:#334de6;font-weight:bold;">● Low</span>'
            '<span style="color:#999;">→</span>'
            '<span style="color:#e63333;font-weight:bold;">● High cliff sensitivity</span>'
            '<span style="color:#999;font-size:0.82em;margin-left:10px;">'
            'Larger circle = selected position</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Position pills selector ───────────────────────────────────────────
    st.pills(
        "Select a position to explore:",
        pill_options,
        key="pos_pills",
        selection_mode="single",
    )

    # Re-read after widget render (user may have just clicked)
    current_pill = st.session_state.get("pos_pills", pill_options[0])
    if current_pill not in pill_options:
        current_pill = pill_options[0]
    selected_rank = pill_options.index(current_pill)
    selected_r = results[selected_rank]

    # ── Detail panel ──────────────────────────────────────────────────────
    label = sensitivity_to_label(selected_r.sensitivity)
    st.subheader(
        f"Position #{selected_rank + 1} — {label} cliff sensitivity "
        f"(SALI {selected_r.sensitivity:.3f} | {selected_r.percentile:.0f}th percentile)"
    )
    st.caption(f"R-group at this position: `{selected_r.rgroup_smiles}`  |  "
               f"Cut bond: atom {selected_r.atom_idx} — atom {selected_r.neighbor_idx}")

    col_features, col_change, col_evidence = st.columns([1, 0.9, 1.6])

    with col_features:
        if selected_r.attribution:
            st.markdown("**What drove this prediction** *(SHAP feature contributions)*")
            st.caption("▲ Red = raises cliff sensitivity · ▼ Blue = lowers cliff sensitivity · bar width = contribution magnitude")
        else:
            st.markdown("**Feature values**")
        _render_attribution(selected_r)

    with col_change:
        st.markdown("**Recommended change types** *(cliff likelihood)*")
        st.caption("Which property modification causes the largest activity swings at this pharmacophore context · bar = predicted |Δ|")
        _render_change_type_recs(selected_r.change_type_recs)

    with col_evidence:
        st.markdown("**Real-world evidence from ChEMBL:**")
        if selected_r.evidence:
            _render_evidence(selected_r.evidence)
        else:
            st.caption("No evidence examples found for this position.")

    # ── Interpretation panel ──────────────────────────────────────────────
    st.divider()
    st.subheader("Interpretation Guide")

    sens_values = [r.sensitivity for r in results]
    spread = max(sens_values) - min(sens_values)

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.metric("Most cliff-prone position", f"SALI {results[0].sensitivity:.3f}")
        st.markdown(f"R-group at this position: `{results[0].rgroup_smiles}`")
    with col_i2:
        st.metric("Position spread", f"{spread:.3f}")
        if spread > 0.03:
            st.markdown("Positions vary **meaningfully** — the model can distinguish cliff-prone vs. stable sites.")
        elif spread > 0.015:
            st.markdown("Moderate variation between positions. The top-ranked position is a reasonable starting point.")
        else:
            st.markdown("Positions are **similar** in predicted cliff sensitivity. Consider other criteria (synthetic accessibility, IP) to prioritize.")

    st.markdown("""
**How to read the results:**
- **SALI sensitivity** is the predicted mean(|delpActivity| / modification size) -- how much potency changes *relative to the size of the structural modification*. Higher values mean small R-group changes cause disproportionately large potency shifts (activity cliffs).
- **Higher SALI** = cliff-prone position = small changes here are likely to reveal steep SAR.
- **Lower SALI** = SAR is flat relative to modification size = modifications here move potency proportionally to their structural size.
- The model captures **cliff propensity**, not direction -- it tells you WHERE small changes have outsized effects, not whether a specific change will increase or decrease potency.

**What makes a position cliff-prone?**
Under SALI normalization, scaffold size alone performs **worse than random** -- the size confound has been removed. The genuine predictors are **ring complexity** (core\_n\_rings, 28%), **electronic character** (Gasteiger charge, 17%), **core size** (18%, with reversed sign), and **solvent exposure** (SASA, 11%). Positions on complex ring systems with distinctive electronic character at the attachment point are most cliff-prone. The ML model achieves NDCG@3 = 0.910 vs 0.791 for the size heuristic -- a genuine ML contribution validated on novel scaffolds (+7.4%) and temporal splits (+12.3%).

**Reading the SHAP attribution chart:**
The red/blue bars show exactly which features pushed the model's prediction above or below its baseline. Unlike the raw model (where core\_n\_heavy dominated), SALI SHAP values reflect a **balanced mix of features** -- ring complexity, electronic character, and steric environment all contribute meaningfully. SHAP values are exact for tree-based models (no approximation).

**Reading the change type recommendations:**
The middle panel ranks which *class* of R-group modification is most likely to cause a large activity change at this position. This is a **weak signal** (Spearman 0.27, ~7% variance explained) -- use it to narrow from 11 axes to 3-4 candidates, not as definitive guidance. All 50 targets show positive correlation (the model never anti-predicts), with kinases showing the strongest signal.

**About the evidence examples:**
Each position shows real matched molecular pairs from ChEMBL where modifications were made at pharmacophore-equivalent positions. **Exact matches** mean the same scaffold core exists in the database. **Similar matches** come from different scaffolds whose attachment point has the same local environment (H-bond donors/acceptors, steric crowding, charge, aromaticity). The delpActivity values are measured, not predicted -- these are real potency changes from real assays. Click any target or compound link to view the full ChEMBL record.
    """)


if __name__ == "__main__":
    main()
