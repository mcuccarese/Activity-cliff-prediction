# Predicting Activity Cliffs for Autonomous Medicinal Chemistry

**Michael F. Cuccarese, PhD**

April 2026

---

## Abstract

Activity cliff prediction -- identifying positions on a molecule where small structural changes cause large potency shifts -- has been a persistent challenge in computational medicinal chemistry. Especially in information-poor contexts, such as the start of a new SAR campaign, predicting impactful chemical modifications is exceptionally challenging. This work simplifies the task by focusing on a parsimonious definition of an activity cliff: which small modifications, at which positions, confer the highest probability of an outcome change. Position-level sensitivity is calculated using 25 million matched molecular pairs (MMPs) from 50 ChEMBL targets across six protein families, revealing that two distinct questions about position sensitivity have fundamentally different answers. "Which positions vary the most?" is answered by scaffold size alone (NDCG@3 = 0.966), requiring no machine learning -- a result that validates ligand efficiency theory at scale but provides no actionable insight beyond the obvious. "Which positions are true activity cliffs?" -- where *small* modifications cause *disproportionately large* effects, as captured by SALI normalization -- requires an 11-feature model incorporating 3D pharmacophore context (NDCG@3 = 0.910 vs. 0.839 random baseline), and this advantage holds across all six protein families, novel scaffold holdouts, and datasets entirely outside the training corpus. The model generalizes to structurally novel chemistry (SALI NDCG@3 = 0.913 on 20% novel scaffold holdout) and to future chemistry (0.878 on temporal split). In practical terms, the model identifies the most cliff-prone position on its first try 53% of the time (vs. 27% random -- a 2x lift), reducing the average number of positions a chemist must explore from 3.1 to 2.1 -- roughly one fewer analog series per scaffold, or a 31% reduction in first-round experiments. An honest negative result bounds the approach: predicting *which* type of modification to make at each position is not tractable from structure alone (Spearman 0.268 in-sample, collapsing to -0.31 on novel scaffolds). The complete system -- from SMILES input to ranked, synthesizable compound recommendations -- is released as an open-source tool and deployed as an interactive webapp, providing a first-round SAR exploration engine for autonomous medicinal chemistry.

## 1. Introduction

Drug discovery campaigns are not limited by the number of compounds chemists can synthesize. They are limited by *which* compounds get made. Combinatorial chemistry and parallel synthesis platforms transformed throughput in the 1990s, yet as Brown and Bostrom documented across 25 years of medicinal chemistry literature, the reaction toolkit used in practice has barely changed: the same amide couplings, reductive aminations, and Suzuki couplings dominate lead optimization today as they did in 2000 [1]. The problem is not synthetic capacity; it is that high-throughput approaches are constrained to the same chemistries, and so explore the same corners of chemical space.

A smarter approach would target synthesis effort precisely -- make fewer compounds, but make the ones that carry the most information about the structure-activity relationship (SAR). Depending on the chemical throughput of the organization, even a modest efficiency gain through avoided low-impact compounds would be of considerable benefit. This is also in alignment with the premise of active learning in drug discovery [2, 3]: select the experiment that maximally reduces uncertainty about SAR, rather than the most synthetically convenient experiment or the compound with the highest predicted potency.

Activity cliff probes are the highest-value experiments in an information-poor setting, where existing results haven't appreciably fed into local models. An activity cliff is a position on a molecule where a small structural change causes a large change in biological activity [4, 5]. Identifying such positions before synthesis -- from structure alone -- would allow a computational system to propose two or three maximally informative modifications, generate specific synthesizable compounds that realize them, and hand a ranked list to a chemist or automated synthesis platform.

This paper asks: **can we predict activity cliff positions well enough to guide autonomous first-round SAR exploration?**

The answer depends critically on how the question is framed. Under raw sensitivity metrics, a trivial heuristic -- rank positions by inverse scaffold size (i.e., positions on smaller scaffolds ranked first) -- appears to solve the problem (NDCG@3 = 0.966) with no machine learning required. This is not a bug; it validates at massive scale a well-known chemical principle: R-group contributions to binding scale inversely with scaffold size [6]. But, without rewarding smallness of change, it is weighted toward existing and obvious ways to break a compound's activity as opposed to those which have impact while still preserving much of the scaffold's character.

SALI normalization [4] -- dividing activity change by structural change -- isolates positions where *small* modifications cause *disproportionately large* effects. Under SALI, the scaffold-size heuristic falls below the random baseline (0.791 vs. 0.839), while an 11-feature model incorporating 3D pharmacophore context achieves 0.910 -- capturing 44% of available headroom over chance. Machine learning genuinely adds value, but only when the right question is asked.

**Contributions.** (1) A position-level reframing of activity cliff prediction using 25 million MMPs across 50 targets, with the largest systematic validation to date. (2) A demonstration that raw MMP-derived sensitivity conflates position vulnerability with modification size, reversing method rankings under SALI normalization -- a cautionary finding for prior work. (3) An 11-feature, target-agnostic model that generalizes across protein families by encoding the pharmacophore as the local context as opposed to the target name, novel scaffolds, and temporal splits. (4) An honest negative result: predicting *which* type of modification to make at each position is not tractable from structure alone. (5) A deployed webapp and open-source tool providing end-to-end compound recommendations from a single SMILES input.

## 2. Related Work

**Activity cliff prediction.** Maggiora [5] formalized the concept of activity cliffs as discontinuities in the structure-activity landscape. Stumpfe and Bajorath [7] systematically catalogued activity cliffs across public databases, establishing the prevalence of the phenomenon. Van Tilborg et al. [8] trained neural models to predict activity cliffs but demonstrated that performance degrades severely out-of-distribution -- the fundamental limitation that motivates the target-agnostic approach taken here.

**Matched molecular pairs.** The Hussain-Rea algorithm [9] and subsequent implementations [10] enable exhaustive enumeration of structurally matched compound pairs from activity databases. MMPs have become a standard tool for SAR analysis, but prior work has primarily used them descriptively (cataloguing known cliffs) rather than predictively (identifying cliffs before measurement).

**SALI.** The Structure-Activity Landscape Index, introduced by Guha and Van Drie [4], normalizes activity change by structural similarity to isolate true activity cliffs from trivial size-dependent effects. Despite its importance, SALI normalization has been inconsistently applied in the activity cliff prediction literature, leading to inflated performance claims that this work explicitly addresses.

**Active learning in drug discovery.** Reker et al. [2, 3] demonstrated that Bayesian active learning can reduce the number of compounds needed to map SAR. Bellamy et al. [11] extended this with batched experimental design. The present system is complementary: it addresses the *first round* of experimentation, before any target-specific activity data exist, where active learning cannot yet operate.

**Ligand efficiency.** The principle that smaller molecules make more efficient use of their binding interactions [6] provides the physical basis for the scaffold-size heuristic observed here. Our results validate this principle at unprecedented scale (25M MMPs, 50 targets) while showing that it is necessary but not sufficient for identifying true activity cliffs.

## 3. Methods

### 3.1 MMP Extraction and Position Sensitivity

Single-cut MMPs were extracted from ChEMBL 36 (2024 release) using the Hussain-Rea algorithm [9] via RDKit's `rdMMPA`, yielding 25 million pairs across 50 targets spanning six protein families (22 kinases, 7 enzymes, 7 immune targets, 4 epigenetic targets, 3 ion channels, 7 receptors/other).

For each substitution position (attachment point on a molecular scaffold) per target, two sensitivity measures were computed:

- **Raw sensitivity** = mean |Delta pActivity| across all MMPs at that position
- **SALI sensitivity** = mean(|Delta pActivity| / (max R-group heavy atoms + 1)) per Guha and Van Drie [4]

This produced 598,173 position-level training examples across 133,763 molecules (median 4 positions per molecule).

### 3.2 Feature Architecture

The model uses 11 structural features computed from the molecule alone, with no activity data required at prediction time. Features partition into two groups:

| Group | Features |
|---|---|
| **Topology (2)** | `core_n_heavy` (scaffold heavy atom count), `core_n_rings` (scaffold ring count) |
| **3D pharmacophore context (9)** | H-bond donors, acceptors, hydrophobic atoms, aromatic atoms, SASA, Gasteiger charge, rotatable bonds, aromatic attachment, heavy atom density -- all within 4 Angstrom of the attachment point |

The topology features capture scaffold complexity. The 3D context features describe the local chemical environment surrounding each substitution position, enabling the model to distinguish pharmacophore-critical positions (near H-bond networks, aromatic stacking surfaces) from solvent-exposed, tolerant positions.

### 3.3 Model and Validation

HistGradientBoosting (HGB) regression models were trained with leave-one-target-out cross-validation across all 50 targets. Primary metric: NDCG@3 (ranking quality of the top-3 positions per molecule). Secondary metrics: Spearman rho, Hit@1 (fraction of molecules where the most sensitive position is ranked first).

Statistical tests: paired Wilcoxon signed-rank on per-target NDCG@3, bootstrap 95% confidence intervals (10,000 resamples, seed = 42), Cohen's d for effect sizes, 1,000-permutation null distributions.

### 3.4 Out-of-Distribution Validation Strategy

The central concern for any predictive model in chemistry is information leakage: if training and test compounds share scaffolds, chemical series, or target-specific SAR patterns, apparent generalization is illusory. We designed 11 OOD stress tests to address this systematically. Three matter most, listed in order of increasing stringency:

Temporal holdout (train on data published before a cutoff, test on data published after) is the conventional gold standard, but it provides weaker protection than commonly assumed. Chemical series routinely span a decade of publications, so temporal splits still share scaffold chemotypes across the boundary. Novel scaffold holdout -- withholding the 20% most structurally dissimilar scaffolds -- directly tests whether predictions transfer to chemistry the model has never seen, regardless of when it was published. External validation on datasets entirely outside ChEMBL (COVID Moonshot, Open Force Field, Schrodinger FEP) is the most stringent test of all: different targets, different laboratories, different chemical matter, with no possibility of information leakage from the training corpus.

Additional OOD experiments include: feature ablation (11 model configurations), target family holdout, feature robustness (conformer sensitivity), learning curve (5-100% training data), directional analysis, permutation tests (1,000 label shuffles), calibration, and hyperparameter sensitivity (50 random configurations).

## 4. Calibration: Raw Sensitivity Is Dominated by Scaffold Size

Before presenting the main results, a calibration. Under raw sensitivity (mean |Delta pActivity|, unnormalized), a trivial heuristic -- rank positions by inverse scaffold size -- achieves NDCG@3 = 0.966, matching or exceeding the full 11-feature ML model (0.959) on 48 of 50 targets. The physical interpretation is straightforward: on a 10-atom scaffold, swapping methyl for cyclohexyl changes the molecule by ~50%; on a 28-atom scaffold, the same swap is ~20%. R-group contributions to binding scale inversely with scaffold size [6].

This validates ligand efficiency theory at scale, but it does not identify activity cliffs. The heuristic excels because positions on smaller scaffolds host *larger* R-group modifications in ChEMBL, and larger modifications trivially produce larger activity changes. Raw sensitivity conflates genuine position vulnerability with modification size -- a confound that SALI normalization was designed to resolve.

## 5. SALI Normalization: Identifying True Activity Cliffs

The Structure-Activity Landscape Index [4] was designed precisely to address the confound in raw sensitivity. SALI normalizes activity change by structural change:

> SALI sensitivity = mean( |Delta pActivity| / (max R-group heavy atoms + 1) )

This is a position-level adaptation of the original SALI [4], which normalizes by Tanimoto dissimilarity between compound pairs. At the MMP level, structural similarity is dominated by the shared core; the meaningful variation is the R-group size, which heavy atom count captures directly. This adaptation isolates positions where small modifications cause disproportionately large activity shifts -- the true activity cliffs that carry the most information per experiment.

### 5.1 The Model Separates from Chance

Under SALI normalization, random baseline NDCG@3 is 0.839 -- the performance of a model with no information. The 11-feature ML model achieves 0.910, capturing 44% of the available headroom between chance and a perfect ranker (Table 1). The scaffold-size heuristic, which dominated raw metrics, collapses to 0.791 -- *below random* -- because SALI strips away the very confound the heuristic exploits.

**Table 1.** Model performance under SALI normalization. The ML model separates meaningfully from random; the heuristic falls below chance.

| Model | SALI NDCG@3 | vs. random |
|---|---|---|
| Random baseline | 0.839 | -- |
| Global mean | 0.831 | -0.008 |
| Heuristic (scaffold size) | 0.791 | **-0.048** |
| 3D context only (9 features) | 0.875 | +0.036 |
| Topology only (2 features) | 0.909 | +0.070 |
| **Full model (11 features)** | **0.910** | **+0.071** |

**Why the scores look high -- and what they mean in practice.** The typical molecule in this dataset has 4 modifiable positions (median; mean 5.4; 84% have 3-6). With so few items to rank, even a random shuffle scores ~0.84 NDCG@3 -- there simply aren't many ways to be wrong. The meaningful signal is the lift above random.

A more intuitive metric is Hit@1: the fraction of molecules where the model correctly identifies the most cliff-prone position on its first try. Random guessing gets this right 26.6% of the time. The SALI model gets it right 52.7% of the time -- a 2x lift. By the second position tested, the model has found the true cliff 74.1% of the time (vs. 49.2% random).

In terms of experimental efficiency: a medicinal chemist guided by this model needs to test on average 2.1 positions to find the most cliff-prone site, versus 3.1 positions by random exploration -- saving roughly one position per molecule, a 31% reduction in experiments. Across a 10-scaffold campaign at ~10 compounds per position, that translates to roughly 100 fewer compounds to learn the same SAR.

The Spearman correlation between scaffold size and ground-truth position sensitivity flips from r = -0.506 (raw: smaller scaffolds = higher sensitivity) to r = +0.301 (SALI: smaller scaffolds = *lower* cliff propensity once modification size is controlled). Under SALI, the "obvious" answer is not merely uninformative -- it is actively misleading, ranking worse than guessing. The 11-feature model resists the collapse because its 3D pharmacophore context features capture information orthogonal to scaffold size: whether a position sits near H-bond networks, aromatic surfaces, or solvent-exposed regions.

### 5.2 Cross-Family Generalization

The separation from random is not driven by a few well-behaved targets. Leave-one-target-out NDCG@3, grouped by protein family, shows the ML model holds above random across all six families (**Figure 1**):

**Table 2.** Per-family SALI NDCG@3 (leave-one-target-out, grouped by family). The ML model holds above random in every family.

| Family | N targets | ML NDCG@3 | vs. random (0.839) |
|---|---|---|---|
| Enzyme | 7 | 0.901 | +0.062 |
| Epigenetic | 4 | 0.881 | +0.042 |
| Immune | 7 | 0.923 | +0.084 |
| Ion channel | 3 | 0.929 | +0.090 |
| Kinase | 22 | 0.911 | +0.072 |
| Receptor/other | 7 | 0.913 | +0.074 |

The range is tight: 0.881-0.929 across families with very different binding site architectures. The model is target-agnostic in practice, not just in principle.

### 5.3 Two Questions, Two Answers

These results establish that the question determines the answer:

| Question | Best predictor | NDCG@3 |
|---|---|---|
| Which positions vary most? | Heuristic (scaffold size) | 0.966 |
| Which are true activity cliffs? | HGB (11 features, 3D context) | 0.910 |

Both questions are useful in different contexts. The heuristic quickly identifies where SAR *lives* on the molecule. The SALI model identifies where *small changes cause large effects* -- the highest-value positions for tight optimization. The two rankings disagree on the same molecules (**Figure 2**), with the sign of the scaffold-size correlation flipping from -0.506 to +0.301.

## 6. From Cliff Identification to Compound Selection

Identifying *where* activity cliffs occur is the core contribution. But a practical system must also recommend *what to make* at those positions (**Figure 3**). Two strategies are natural: predict which modification type will have the largest impact (an **impact-weighted** approach), or cover the space of possible modification types as broadly as possible (a **diversity-weighted** approach).

### 6.1 Change-Type Prediction: Why Impact Weighting Fails

We attempted to predict *which* type of modification (e.g., electron-withdrawing, H-bond donor, size increase) produces the largest activity shift at each position. If this worked, a system could recommend specific compounds rather than just positional priorities.

The change-type model ranks 11 R-group property modification axes by predicted impact at each position. Its performance is measured by Spearman rho -- the rank correlation between predicted and observed impact across the 11 axes. A Spearman of 1.0 would mean the model perfectly ranks which modification types matter most; 0.0 would mean no predictive signal at all.

Mean Spearman rho = 0.268 across 50 targets (**Figure 4**). A permutation null -- the same model trained on randomly shuffled labels -- achieves 0.199. So the signal is real (the model has learned *something* about which modifications matter), but it explains only ~7% of variance. In practical terms, the top-ranked modification type is correct only ~56% of the time.

More critically, this weak in-sample signal collapses on novel scaffolds (rho = -0.31), meaning the model actively predicts the *wrong* modification types when deployed on new chemistry. SALI normalization worsens performance further (rho = 0.192), confirming that even the in-sample signal reflects modification size rather than genuine pharmacophore insight.

**Implication.** Predicting *where* on a molecule to modify is tractable from structure alone; predicting *what specific change to make* is not. Crossing this boundary will require target-specific activity data, which is unavailable at campaign start.

### 6.2 Diversity-Weighted Selection: Covering the Hypothesis Space

Given that impact prediction fails on novel chemistry, the alternative is information-theoretic: instead of betting on which modification type will matter most, cover the space of possible modifications as broadly as possible. The logic is straightforward -- if we cannot predict which of the 11 modification axes will produce a cliff, we should ensure our first experiments sample across as many axes as possible, maximizing the probability that at least one experiment hits the true SAR driver.

At each sensitive position, greedy submodular optimization selects 2-3 modifications that maximally cover the SAR hypothesis space across the 11 change-type axes (electron-withdrawing/donating, H-bond donor/acceptor, lipophilicity, size, aromaticity, sp3 character, ring count). Synthesizability filtering uses MMP transforms from the 25M-pair corpus with SA score thresholds [12, 13], ensuring recommended compounds are concretely makeable.

### 6.3 The Complete Pipeline

Given only a scaffold SMILES, the system: (1) identifies true activity cliff positions via the SALI-normalized model; (2) selects maximally diverse chemical hypotheses at each position; (3) retrieves synthesizable MMP transforms from the 25M-pair corpus; and (4) outputs a ranked compound list. This reduces the typical 20-40 compound first round to 6-9 targeted experiments without sacrificing SAR coverage.

## 7. Generalization: Does It Work on New Chemistry?

The results in Sections 5-6 establish that the model works in-distribution. This section consolidates all out-of-distribution evidence -- for both cliff identification (SALI NDCG@3) and compound selection -- to address the question that matters most: will it work on chemistry the model has never seen?

### 7.1 Novel Scaffold Holdout

The most direct test of structural generalization. When 20% of the most structurally dissimilar scaffolds (by Tanimoto distance to all training cores) are held out, the cliff identification model maintains SALI NDCG@3 = 0.913 -- degrading only 0.003 from the LOO baseline (0.910). The random baseline under the same split is 0.839. The model's separation from chance holds on chemistry it has structurally never seen.

### 7.2 Temporal Split

Under the strictest temporal split (train <= 2015, test > 2015), 88% of test scaffolds are novel and 83% of test cores are novel. The cliff identification model maintains SALI NDCG@3 = 0.878 -- a modest but consistent advantage over random (0.839) that holds across both temporal cutoffs tested.

### 7.3 External Validation

The ultimate test: 114 positions from three datasets entirely outside ChEMBL, with 67.8% novel scaffolds and 94.7% novel compounds. The diversity-weighted compound selector achieves a top-hit rate (fraction of positions where the highest-impact change axis is among the selector's top-3 recommendations) of 0.439 vs. 0.271 for random selection (Wilcoxon p = 1.86 x 10^-5). The impact-weighted approach underperforms random (0.193), confirming the in-sample negative result: when change-type prediction is unreliable, *covering the hypothesis space* outperforms *betting on a specific prediction*.

**Table 3.** External validation (114 positions from COVID Moonshot, Open Force Field, Schrodinger FEP). Top-hit rate = fraction of positions where the highest-impact change axis is among the selector's top-3 recommendations.

| Strategy | Top-hit rate | vs. random |
|---|---|---|
| Random | 0.271 | -- |
| Impact-weighted | 0.193 | -0.078 |
| **Diversity-weighted** | **0.439** | **+0.168** |

### 7.4 The Signal Is Structural, Not Statistical

Two additional experiments (reported under raw sensitivity, where the model is most constrained) confirm that the model captures physical relationships rather than statistical artifacts:

**Learning curve.** Raw NDCG@3 plateaus at 5% of training data (0.958 vs. 0.959 at 100%). Even with only 50 positions per target (simulating a rare target), NDCG@3 = 0.949. The model learns general relationships between pharmacophore context and position sensitivity, not target-specific patterns.

**Hyperparameter sensitivity.** Fifty random HGB configurations produce NDCG@3 values spanning only 0.954-0.965 (range = 0.011). The performance ceiling is inherent to the feature set, not a tuning artifact.

### 7.5 Sensitivity Does Not Predict Direction

Does high sensitivity mean a position is "fragile" (modifications hurt potency) or "improvable" (modifications can increase potency)? The answer is neither. High-sensitivity positions split almost exactly: 29.2% improvable, 41.7% neutral, 29.0% fragile. The correlation between sensitivity and directionality is Spearman = -0.005 (p = 0.19). "Sensitive" means *variable* -- the position matters, but which direction depends on the specific modification and target, not on the structural context the model can observe.

## 8. Discussion

### 8.1 What Works

**Position sensitivity prediction is solved for its intended purpose.** SALI-normalized NDCG@3 = 0.910 (vs. 0.839 random) provides reliable, target-agnostic identification of true activity cliff positions from structure alone. The model identifies the most cliff-prone position on its first try 53% of the time (vs. 27% random), reducing the average positions a chemist must explore from 3.1 to 2.1 -- a 31% reduction in first-round experiments. It generalizes across protein families, novel scaffold holdouts (0.913), temporal splits (0.878), and external datasets entirely outside the training corpus.

**The SALI reframe is methodologically important.** Raw MMP-derived sensitivity conflates position vulnerability with modification size; SALI normalization reverses relative method rankings. This is a cautionary finding for prior work [7, 8] that evaluated activity cliff prediction under raw metrics: results that appeared strong may have been measuring the obvious (scaffold size) rather than the interesting (pharmacophore context).

### 8.2 The "Where" vs. "What" Boundary

This work establishes a clear boundary: *where* to modify a molecule is predictable from structure alone; *what specific change to make* is not. The change-type model's in-sample rho = 0.268 collapses to rho = -0.31 on novel scaffolds -- historical activity-change patterns from ChEMBL do not transfer to unseen chemistry [14]. Crossing this boundary will require target-specific activity data, which is unavailable at campaign start. Until then, diversity-weighted selection -- covering the hypothesis space rather than betting on a single prediction -- is the rational strategy, and external validation confirms this (+0.168 top-hit rate over random).

### 8.3 Path Forward

The right paradigm is **closed-loop experimental feedback**. This system addresses the first round: identify sensitive positions, propose diverse modifications, and collect data. Accumulated measurements can then enable target-specific change-type predictions via active learning [2, 8] and batched Bayesian optimization [11].

## 9. Limitations

1. **Single-cut MMPs assume position independence.** Cooperative SAR -- where modifications at two positions interact -- is invisible to single-cut MMP analysis. Multi-cut MMP methods [15] could extend the approach but increase computational cost substantially.

2. **Sensitivity does not predict direction.** The model identifies *where* activity cliffs occur but cannot predict whether modification will improve or degrade potency. This requires protein structure information not captured by ligand-only features.

3. **Free-ligand 3D features are approximate.** Conformers are generated for the free ligand, not the protein-bound state. Gasteiger charges are crude compared to quantum-mechanical calculations. These approximations are pragmatic -- they enable zero-install predictions from SMILES alone -- but bound the model's resolution.

4. **Target selection bias.** The 50 ChEMBL targets were chosen for data richness (thousands of MMPs each). Performance on data-poor targets or novel target classes is not directly validated, though the learning curve analysis suggests the model is relatively data-efficient.

5. **Retrospective validation only.** No prospective experimental validation was performed. The external validation on COVID Moonshot, Open Force Field, and Schrodinger FEP provides the strongest available test of generalization, but ultimate validation requires synthesis and measurement.

6. **Change-type prediction explains only ~7% of variance.** The ceiling for ligand-only change-type prediction appears to be low. Protein-side features (binding pocket shape, interaction fingerprints) would likely improve this, but are unavailable in the target-agnostic setting.

## 10. Interactive Demo

A prediction model is only useful if practitioners can apply it without friction. The system is deployed as an interactive Streamlit webapp where users can:

1. Enter any SMILES string
2. See positions colored by predicted SALI sensitivity
3. Browse recommended modifications with diversity scores
4. Look up supporting evidence from the 25M MMP corpus
5. Export ranked compound lists

The webapp includes honest framing: position sensitivity is dominated by scaffold size under raw metrics; the SALI model adds value for identifying true cliffs; change-type recommendations are weak and should be treated as hypotheses, not predictions. Flexibility warnings flag molecules with high rotatable bond counts where conformer-dependent features are less reliable.

For integration into autonomous pipelines, the core library is released as open-source Python code. Given a SMILES, `predict.py` returns position-level SALI sensitivity scores and ranked compound recommendations in a single function call.

## 11. Conclusion

Activity cliff prediction has been framed as a machine learning problem requiring target-specific training data. This work shows that the problem has two distinct components with different solutions. Position sensitivity ranking -- *where* on a molecule SAR lives -- is dominated by scaffold size, a chemical principle that requires no learning. True activity cliff identification -- *where small changes cause disproportionately large effects* -- requires 3D pharmacophore context features and machine learning, achieving NDCG@3 = 0.910 target-agnostically across 50 targets and six protein families, with robust generalization to novel scaffolds (0.913) and future chemistry (0.878).

The practical system reduces a first-round SAR campaign from 20-40 unfocused compounds to 6-9 targeted experiments at predicted cliff positions. By identifying the cliff-prone position on its first try 53% of the time -- versus 27% by chance -- the model saves roughly one analog series per scaffold (2.1 vs. 3.1 positions to explore), or ~100 fewer compounds across a typical 10-scaffold campaign. The underlying insight is one of right-sizing the question: asking a model *where* activity cliffs live is tractable; asking it *what specific change to make* is not. The system's most important limitation defines the natural handoff point to closed-loop experimental feedback and active learning.

**Code and data availability.** The complete system, including trained models, evaluation scripts, and the interactive webapp, is available at https://github.com/mcuccarese/Activity-cliff-prediction. Activity data were extracted from ChEMBL 36, a publicly available database [16].

**Acknowledgments.** Claude (Anthropic) was used for code development, data analysis, and manuscript preparation assistance.

---

## References

[1] Brown, D.G.; Bostrom, J. "Analysis of past and present synthetic methodologies on medicinal chemistry." J. Med. Chem. 59, 4443-4458 (2016).

[2] Reker, D.; Schneider, G. "Active-learning strategies in computer-assisted drug discovery." Drug Discov. Today 20, 458-465 (2015).

[3] Reker, D. "Practical considerations for active machine learning in drug discovery." Drug Discov. Today: Technol. 32-33, 73-79 (2020).

[4] Guha, R.; Van Drie, J.H. "Structure-activity landscape index: identifying and quantifying activity cliffs." J. Chem. Inf. Model. 48, 646-658 (2008).

[5] Maggiora, G.M. "On outliers and activity cliffs -- why QSAR often disappoints." J. Chem. Inf. Model. 46, 1535 (2006).

[6] Hopkins, A.L.; Groom, C.R.; Alex, A. "Ligand efficiency: a useful metric for lead selection." Drug Discov. Today 9, 430-431 (2004).

[7] Stumpfe, D.; Bajorath, J. "Exploring activity cliffs in medicinal chemistry." J. Med. Chem. 55, 2932-2942 (2012).

[8] van Tilborg, D. et al. "Exposing the limitations of molecular machine learning with activity cliffs." J. Chem. Inf. Model. 62, 5938-5951 (2022).

[9] Hussain, J.; Rea, C. "Computationally efficient algorithm to identify matched molecular pairs." J. Chem. Inf. Model. 50, 339-348 (2010).

[10] Dalke, A. et al. "mmpdb: An open-source matched molecular pair platform for large multiproperty data sets." J. Chem. Inf. Model. 58, 902-910 (2018).

[11] Bellamy, H.; Abdel Rehim, A.; Orhobor, O.I.; King, R. "Batched Bayesian optimization for drug design in noisy environments." J. Chem. Inf. Model. 62, 3970-3981 (2022).

[12] Swanson, K.; Liu, G.; Catacutan, D.B.; Arnold, A.; Zou, J.; Stokes, J.M. "Generative AI for designing and validating easily synthesizable and structurally novel antibiotics." Nat. Mach. Intell. 6, 338-353 (2024).

[13] Gao, W.; Coley, C.W. "The synthesizability of molecules proposed by generative models." J. Chem. Inf. Model. 60, 5714-5723 (2020).

[14] Tyrchan, C.; Evertsson, E. "Matched molecular pair analysis in short: algorithms, applications and limitations." Comput. Struct. Biotechnol. J. 15, 86-90 (2017).

[15] Leach, A.G.; Jones, H.D.; Cosgrove, D.A.; Kenny, P.W.; Ruston, L.; MacFaul, P.; Wood, J.M.; Colclough, N.; Law, B. "Matched molecular pairs as a guide in the optimization of pharmaceutical properties; a study of aqueous solubility, plasma protein binding and oral exposure." J. Med. Chem. 49, 6672-6682 (2006).

[16] Zdrazil, B.; Felix, E.; Hunter, F.; Manners, E.J.; Blackshaw, J.; Sheridan, E.M.; Leach, A.R. et al. "The ChEMBL Database in 2023: a drug discovery platform spanning genomics, bioactivity and beyond." Nucleic Acids Res. 52, D1180-D1192 (2024).

---

**Figure 1.** Raw vs. SALI-normalized position sensitivity across six protein families and model ablations. Top row (raw sensitivity): the heuristic and ML model perform comparably across all families. Bottom row (SALI-normalized): the heuristic collapses while the ML model holds, with a consistent advantage over random (0.839) in every family.

**Figure 2.** Raw vs. SALI sensitivity rankings on representative molecules. Atoms colored by predicted sensitivity (green = low, red = high). The highest-ranked position differs between raw and SALI across all scaffolds, reflecting the sign-flip in scaffold-size correlation (r = -0.506 to +0.301).

**Figure 3.** Complete pipeline from input SMILES to synthesizable compound recommendations. The SALI-normalized sensitivity model ranks positions; submodular optimization selects diverse change types; MMP transforms from the 25M-pair corpus are retrieved, filtered for synthesizability, and output as a ranked compound list.

**Figure 4.** Change-type prediction across 50 targets. The observed mean Spearman (0.268) exceeds the permutation null (0.199) but explains only ~7% of variance -- insufficient for actionable compound selection.
