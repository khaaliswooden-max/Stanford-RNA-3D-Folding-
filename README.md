# 🧬 Stanford RNA 3D Folding Competition - Part 2

Predict C1' atom 3D coordinates for RNA sequences using a multi-phase approach.

## 📁 Project Structure

```
├── phase1_data_exploration.py    # Data analysis & baseline
├── phase2_template_prediction.py # MMseqs2 template matching
├── phase3_deep_learning.py       # RhoFold/ESMFold integration
├── phase4_refinement.py          # Energy minimization
├── phase5_ensemble_submission.py # Final ensemble pipeline
├── rna-baseline-simple.ipynb     # Simple Kaggle notebook
├── stanford-rna-3d-submission.ipynb # Full submission notebook
└── README.md
```

## 🚀 Quick Start

```bash
# Phase 1: Explore data and create baseline
python phase1_data_exploration.py

# Phase 2: Template-based prediction
python phase2_template_prediction.py

# Phase 3: Deep learning prediction
python phase3_deep_learning.py

# Phase 4: Structure refinement
python phase4_refinement.py

# Phase 5: Create final ensemble submission
python phase5_ensemble_submission.py
```

## 📊 Phase Overview

### Phase 1: Data Exploration & Baseline
- Load and analyze competition data
- Visualize sequence distributions
- Create A-form helix baseline submission
- Validate submission format

### Phase 2: Template-Based Prediction
- MMseqs2 sequence similarity search
- Extract C1' coordinates from PDB structures
- Handle temporal cutoff filtering
- Map alignments to coordinates

### Phase 3: Deep Learning Integration
- RhoFold+ for RNA structure prediction
- ESMFold as alternative predictor
- Ensemble multiple prediction methods
- Handle long sequences

### Phase 4: Structure Refinement
- Secondary structure prediction (ViennaRNA/Nussinov)
- Base pairing constraints
- Energy minimization
- Clash resolution

### Phase 5: Final Ensemble
- Combine all phase predictions
- Model ranking and selection
- Clustering for diversity
- Final submission generation

## 🔧 Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
biopython
```

Optional (for deep learning):
```
torch
rhofold
esm
viennarna
```

## 📈 Expected Performance

| Method | MAE (Å) |
|--------|---------|
| A-form baseline | ~30+ |
| Template-based | ~15-25 |
| RhoFold | ~10-20 |
| Ensemble | ~8-15 |

## 🏆 Competition Details

- **Task**: Predict 3D coordinates of C1' atoms
- **Metric**: Mean Absolute Error (MAE)
- **Models**: 5 structure predictions per sequence
- **Format**: CSV with x/y/z coordinates for each model

## 📝 Notes

- Temporal cutoffs must be respected for template methods
- Long sequences (>1000 nt) may need special handling
- GPU recommended for deep learning phases
- ViennaRNA optional but improves refinement

## 🔗 Resources

- [Kaggle Competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- [RhoFold Paper](https://arxiv.org/abs/2207.01586)
- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/)
