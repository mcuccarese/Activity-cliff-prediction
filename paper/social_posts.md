# Social Media Posts — Activity Cliffs Launch

---

## LinkedIn

---

Activity cliff prediction has been a grail of computational medicinal chemistry for 15+ years. And it has consistently failed to deliver when you actually try to use it on new chemistry.

We took a step back and asked a simpler question. Instead of "predict whether this compound pair is an activity cliff" (which requires target-specific training data you don't have yet), we asked: "which positions on this molecule are most likely to be activity cliffs?" — positions where small modifications cause disproportionately large potency shifts.

We built a dataset of 25 million matched molecular pairs across 50 ChEMBL targets and found something surprising. Under raw sensitivity metrics, a trivial heuristic — just rank positions by inverse scaffold size — gets NDCG@3 = 0.966. No ML needed. Smaller scaffolds have more sensitive positions. This is ligand efficiency theory, validated at massive scale.

But here's the catch: that heuristic doesn't find real activity cliffs. It finds positions where big R-group swaps cause big potency changes — which is obvious and unhelpful. When you normalize for modification size (SALI normalization), the heuristic collapses to *below random*. An 11-feature model using 3D pharmacophore context achieves NDCG@3 = 0.910 — and this holds across all six protein families, novel scaffolds, and external datasets the model has never seen.

The practical upshot: given only a SMILES string, the system identifies where activity cliffs will occur and proposes 6–9 diverse, synthesizable compounds for first-round SAR exploration. On external validation (COVID Moonshot, Schrodinger FEP), diversity-driven selection achieves a 0.439 top-hit rate vs. 0.271 random.

Two honest findings that I think strengthen the work:
1. Predicting *what specific modification to make* is not tractable from structure alone (Spearman 0.268, collapses to -0.31 on novel scaffolds). The model knows *where* to look but not *what* to try.
2. When change-type prediction fails, covering the hypothesis space with diverse modifications beats betting on any specific prediction.

The system is deployed as an interactive webapp and open-source tool. Enter a SMILES, get position-level sensitivity maps and ranked compound recommendations.

Webapp: https://activity-cliffs-5gnirhr3k3ybhwhz7de7ua.streamlit.app/
GitHub: https://github.com/mcuccarese/Activity-cliff-prediction
Preprint: [arXiv link]

#MedicinalChemistry #DrugDiscovery #Cheminformatics #MachineLearning #SAR #ActivityCliffs

---

## X (Twitter) — Thread

---

**Tweet 1 (hook):**
Activity cliff prediction has been a 15-year problem in computational med chem. We think we cracked where the field went wrong.

The trick: stop predicting compound-pair cliffs. Start predicting position-level cliffs. And normalize for modification size.

25M matched molecular pairs, 50 targets. Here's what we found.

**Tweet 2 (the surprising baseline):**
A trivial heuristic — rank positions by inverse scaffold size — gets NDCG@3 = 0.966.

No ML. No features. Just "smaller scaffolds = more sensitive positions."

This validates ligand efficiency theory at massive scale. But it does NOT find real activity cliffs.

**Tweet 3 (the fix):**
Why? Raw sensitivity conflates position vulnerability with modification size. Bigger R-groups on small scaffolds = bigger activity changes. Obvious.

SALI normalization (divide activity change by structural change) strips this away. The heuristic collapses to BELOW random (0.791 vs 0.839).

**Tweet 4 (what works):**
An 11-feature model with 3D pharmacophore context achieves NDCG@3 = 0.910 under SALI.

It holds across 6 protein families, novel scaffolds, temporal splits, and external datasets (COVID Moonshot, Schrodinger FEP).

ML adds value — but only when you ask the right question.

**Tweet 5 (the honest negative):**
What doesn't work: predicting WHAT modification to make (Spearman 0.268, collapses to -0.31 on new chemistry).

The model knows WHERE to look but not WHAT to try. So we default to information-theoretic diversity — cover the hypothesis space, don't bet on one answer.

**Tweet 6 (CTA):**
Deployed webapp — enter any SMILES, get position sensitivity maps + ranked compound recommendations:

Webapp: https://activity-cliffs-5gnirhr3k3ybhwhz7de7ua.streamlit.app/
GitHub: https://github.com/mcuccarese/Activity-cliff-prediction
Preprint: [arXiv link]
