# Predicting Activity Cliffs for Autonomous Medicinal Chemistry

Code, trained models, and interactive webapp for the preprint:

> **Predicting Activity Cliffs for Autonomous Medicinal Chemistry**
> Michael F. Cuccarese, PhD
> [arXiv link]

**Try it now:** [Interactive Webapp](https://activity-cliffs-5gnirhr3k3ybhwhz7de7ua.streamlit.app/) -- enter any SMILES, get position-level sensitivity maps and compound recommendations.

## Key Results

| Question | Best predictor | NDCG@3 |
|---|---|---|
| Which positions vary most? | Scaffold size heuristic | 0.966 |
| **Which are true activity cliffs?** | **11-feature ML model (3D pharmacophore context)** | **0.910** |

- **25 million matched molecular pairs** from ChEMBL 36 across 50 targets and 6 protein families
- SALI normalization isolates true cliffs (small change, large effect) from trivial size artifacts
- The scaffold-size heuristic falls **below random** (0.791 vs. 0.839) under SALI -- the "obvious" answer is actively misleading
- The model identifies the cliff-prone position on its first try **53% of the time** (vs. 27% random -- a 2x lift), saving ~1 analog series per scaffold
- External validation on COVID Moonshot, Open Force Field, and Schrodinger FEP: **0.439 top-hit rate** (vs. 0.271 random)

## Quick Start

### Use the prediction API (no ChEMBL needed)

```python
from webapp.predict import predict_sensitivity

results = predict_sensitivity("c1ccc(NC(=O)c2ccccc2)cc1")
for pos in results:
    print(f"Position {pos.atom_idx}: sensitivity={pos.sensitivity:.3f} (percentile={pos.percentile:.0f}%)")
```

### Run the webapp locally

```bash
conda create -n activity-cliffs python=3.11
conda activate activity-cliffs
conda install -c conda-forge rdkit
pip install -r requirements.txt

streamlit run webapp/app.py
```

### Reproduce from ChEMBL

Download [ChEMBL 36 SQLite](https://www.ebi.ac.uk/chembl/) and set `CHEMBL_SQLITE_PATH`:

```bash
export CHEMBL_SQLITE_PATH="/path/to/chembl_36.db"

# Extract matched molecular pairs
python scripts/list_targets.py --top 50
python scripts/extract_mmps.py --target CHEMBL204

# Compute features
python scripts/compute_mmp_features.py --target CHEMBL204
python scripts/compute_3d_context.py --target CHEMBL204

# Train models
python scripts/prepare_position_data.py
python scripts/train_final_model.py
```

## Repository Structure

```
src/activity_cliffs/        Core library
  data/                     ChEMBL curation, MMP extraction
  features/                 3D pharmacophore context, property change vectors
  models/                   HistGradientBoosting position & change-type models
  recommender/              Compound selection and ranking

scripts/                    Reproducible analysis pipeline
  ood/                      Out-of-distribution stress tests
  external_validation/      COVID Moonshot, OpenFE, Schrodinger FEP

webapp/                     Streamlit interactive explorer
  app.py                    Main UI
  predict.py                Prediction API (SMILES in, sensitivity out)
  model/                    Model metadata and distributions

paper/                      Preprint (markdown + LaTeX + figures)
```

## Model Details

**Position sensitivity model:** HistGradientBoosting regressor trained on 598,173 position-level examples from 25M MMPs across 50 ChEMBL targets. 11 features: 2 topological (scaffold size, ring count) + 9 3D pharmacophore context (H-bond donors/acceptors, hydrophobic atoms, aromatic atoms, SASA, Gasteiger charge, rotatable bonds, aromatic attachment, heavy atom density -- all within 4 Angstrom of the attachment point).

**Validation:** Leave-one-target-out cross-validation. SALI-normalized NDCG@3 = 0.910 (vs. 0.839 random). Holds across 6 protein families, novel scaffold holdouts (0.913), temporal splits (0.878), and external datasets.

**Change-type model:** Mean Spearman rho = 0.268 (statistically significant but practically weak). Collapses on novel scaffolds (rho = -0.31). The system defaults to diversity-driven selection when change-type prediction is unreliable.

## Dependencies

- **RDKit** (conda-forge) -- molecular structure, MMP fragmentation, 3D conformers
- **scikit-learn** -- HistGradientBoosting, evaluation metrics
- **pandas / numpy / scipy** -- data processing
- **Streamlit** -- interactive webapp
- **matplotlib** -- figure generation

See [requirements.txt](requirements.txt) for full specification.

## Citation

```bibtex
@article{cuccarese2026activitycliffs,
  title={Predicting Activity Cliffs for Autonomous Medicinal Chemistry},
  author={Cuccarese, Michael F.},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
