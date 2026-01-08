#!/usr/bin/env python3
"""
Phase 5: Ensemble & Final Submission Pipeline
Stanford RNA 3D Folding Competition - Part 2

Final pipeline combining all phases for optimal submission.
"""

import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

KAGGLE_MODE = os.path.exists('/kaggle/input')

CONFIG = {
    'data_dir': '/kaggle/input/stanford-rna-3d-folding-2/' if KAGGLE_MODE else './data/',
    'num_models': 5,
    'ensemble_weights': {
        'template': 0.4,
        'deep_learning': 0.4,
        'refined': 0.2
    }
}

A_FORM_PARAMS = {'rise': 2.8, 'twist': 32.7, 'radius': 9.0}


# =============================================================================
# Utility Functions
# =============================================================================

def generate_aform_helix(sequence, variation_idx=0):
    """Generate A-form helix coordinates."""
    np.random.seed(variation_idx * 42)
    rise = A_FORM_PARAMS['rise'] * (0.9 + 0.2 * np.random.random())
    twist = np.deg2rad(A_FORM_PARAMS['twist'] * (0.9 + 0.2 * np.random.random()))
    radius = A_FORM_PARAMS['radius'] * (0.9 + 0.2 * np.random.random())
    
    coords = np.array([[radius * np.cos(i * twist), 
                        radius * np.sin(i * twist), 
                        i * rise] for i in range(len(sequence))])
    
    if variation_idx > 0:
        rot = Rotation.from_euler('xyz', np.random.uniform(-30, 30, 3), degrees=True)
        coords = rot.apply(coords)
    
    return coords - np.mean(coords, axis=0)


def compute_rmsd(coords1, coords2):
    """Compute RMSD between coordinate sets."""
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))


def superpose(ref, mobile):
    """Kabsch alignment of mobile onto ref."""
    ref_c = ref - np.mean(ref, axis=0)
    mob_c = mobile - np.mean(mobile, axis=0)
    
    H = mob_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    return (mobile - np.mean(mobile, axis=0)) @ R + np.mean(ref, axis=0)


# =============================================================================
# Model Selection & Ranking
# =============================================================================

def rank_models(models, sequence):
    """Rank models by quality metrics."""
    scores = []
    
    for i, coords in enumerate(models):
        score = 0
        
        # 1. Bond length regularity
        bond_lengths = [np.linalg.norm(coords[j+1] - coords[j]) for j in range(len(coords)-1)]
        bond_std = np.std(bond_lengths)
        score -= bond_std * 10
        
        # 2. No clashes
        dists = pdist(coords)
        clashes = np.sum(dists < 3.0)
        score -= clashes * 50
        
        # 3. Compact structure (radius of gyration)
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
        expected_rg = len(sequence) ** 0.5 * 3  # Rough estimate
        score -= abs(rg - expected_rg) * 0.5
        
        scores.append((i, score))
    
    # Sort by score (higher is better)
    scores.sort(key=lambda x: x[1], reverse=True)
    return [models[i] for i, _ in scores]


def cluster_models(models, rmsd_cutoff=5.0):
    """Cluster similar models and select representatives."""
    if len(models) <= 5:
        return models
    
    # Superpose all to first
    aligned = [models[0]]
    for m in models[1:]:
        aligned.append(superpose(models[0], m))
    
    # Compute pairwise RMSD
    n = len(aligned)
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            rmsd_matrix[i, j] = compute_rmsd(aligned[i], aligned[j])
            rmsd_matrix[j, i] = rmsd_matrix[i, j]
    
    # Greedy clustering
    selected = [0]
    for i in range(1, n):
        min_rmsd = min(rmsd_matrix[i, s] for s in selected)
        if min_rmsd > rmsd_cutoff and len(selected) < 5:
            selected.append(i)
    
    return [models[i] for i in selected]


# =============================================================================
# Ensemble Pipeline
# =============================================================================

class EnsemblePipeline:
    """Final ensemble submission pipeline."""
    
    def __init__(self):
        print("=" * 60)
        print("ENSEMBLE SUBMISSION PIPELINE")
        print("=" * 60)
    
    def load_phase_predictions(self):
        """Load predictions from previous phases."""
        predictions = {}
        
        # Try loading phase outputs
        phase_files = [
            'submission_phase2_template.csv',
            'submission_phase3_deeplearning.csv', 
            'submission_phase4_refined.csv'
        ]
        
        for pf in phase_files:
            if os.path.exists(pf):
                df = pd.read_csv(pf)
                predictions[pf] = self._parse_submission(df)
                print(f"Loaded {pf}")
        
        return predictions
    
    def _parse_submission(self, df):
        """Parse submission CSV to coordinate dict."""
        coords = {}
        
        for _, row in df.iterrows():
            parts = row['ID'].rsplit('_', 1)
            target = parts[0]
            resid = int(parts[1]) - 1
            
            if target not in coords:
                coords[target] = {i: [] for i in range(5)}
            
            for model in range(5):
                coords[target][model].append([
                    row[f'x_{model+1}'],
                    row[f'y_{model+1}'],
                    row[f'z_{model+1}']
                ])
        
        # Convert to numpy
        for target in coords:
            for model in range(5):
                coords[target][model] = np.array(coords[target][model])
        
        return coords
    
    def combine_predictions(self, test_df, phase_predictions):
        """Combine predictions from all phases."""
        print("\nCombining predictions...")
        
        final_predictions = {}
        
        for _, row in test_df.iterrows():
            target = row['target_id']
            sequence = row['sequence']
            
            # Collect all models for this target
            all_models = []
            
            for phase_name, phase_coords in phase_predictions.items():
                if target in phase_coords:
                    for model_idx in range(5):
                        all_models.append(phase_coords[target][model_idx])
            
            # Generate fallbacks if needed
            if len(all_models) < 5:
                for i in range(5 - len(all_models)):
                    all_models.append(generate_aform_helix(sequence, i))
            
            # Rank and select best models
            ranked = rank_models(all_models, sequence)
            
            # Cluster to get diverse set
            selected = cluster_models(ranked[:10])[:5]
            
            # Ensure exactly 5 models
            while len(selected) < 5:
                selected.append(generate_aform_helix(sequence, len(selected)))
            
            final_predictions[target] = selected[:5]
        
        return final_predictions
    
    def create_submission(self, test_df, predictions, output_file='submission.csv'):
        """Create final submission."""
        print("\nCreating final submission...")
        
        rows = []
        
        for _, row in test_df.iterrows():
            target = row['target_id']
            sequence = row['sequence']
            
            models = predictions.get(target, 
                [generate_aform_helix(sequence, i) for i in range(5)])
            
            for i in range(len(sequence)):
                row_data = {
                    'ID': f'{target}_{i+1}',
                    'resname': sequence[i],
                    'resid': i + 1,
                }
                
                for m in range(5):
                    row_data[f'x_{m+1}'] = float(models[m][i, 0])
                    row_data[f'y_{m+1}'] = float(models[m][i, 1])
                    row_data[f'z_{m+1}'] = float(models[m][i, 2])
                
                rows.append(row_data)
        
        df = pd.DataFrame(rows)
        
        cols = ['ID', 'resname', 'resid']
        for i in range(1, 6):
            cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        
        df = df[cols]
        df.to_csv(output_file, index=False, float_format='%.3f')
        
        print(f"Saved {output_file} ({df.shape})")
        return df
    
    def validate(self, submission_df, sample_path):
        """Validate submission."""
        sample = pd.read_csv(sample_path)
        
        print("\nValidation:")
        print(f"   Shape match: {submission_df.shape == sample.shape}")
        print(f"   ID match: {(submission_df['ID'] == sample['ID']).all()}")
        print(f"   NaN count: {submission_df.isna().sum().sum()}")


# =============================================================================
# Main Execution
# =============================================================================

def run_phase5():
    """Execute Phase 5: Final Ensemble Submission."""
    print("\n" + "=" * 60)
    print("PHASE 5: ENSEMBLE & FINAL SUBMISSION")
    print("=" * 60)
    
    pipeline = EnsemblePipeline()
    
    # Load data
    test_df = pd.read_csv(CONFIG['data_dir'] + 'test_sequences.csv')
    print(f"\nLoaded {len(test_df)} sequences")
    
    # Load phase predictions
    phase_preds = pipeline.load_phase_predictions()
    
    # Combine predictions
    final_preds = pipeline.combine_predictions(test_df, phase_preds)
    
    # Create submission
    submission_df = pipeline.create_submission(test_df, final_preds)
    
    # Validate
    pipeline.validate(submission_df, CONFIG['data_dir'] + 'sample_submission.csv')
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("""
    Final submission ready for upload to Kaggle.
    
    Files generated:
    • submission.csv - Final ensemble submission
    
    Next: Upload to competition page for scoring.
    """)
    
    return submission_df


if __name__ == '__main__':
    submission_df = run_phase5()
