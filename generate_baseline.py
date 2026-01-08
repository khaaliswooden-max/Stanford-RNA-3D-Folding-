#!/usr/bin/env python3
"""
Stanford RNA 3D Folding Part 2 - Baseline Submission Generator
Generates A-form helix coordinates as a baseline submission.
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

# A-form RNA helix parameters (Angstroms)
A_FORM_PARAMS = {
    'rise': 2.8,      # Rise per residue along helix axis
    'twist': 32.7,    # Twist angle per residue (degrees)
    'radius': 9.0     # Radius of helix
}

def generate_helix_coords(sequence, variation_idx=0):
    """Generate A-form helix C1' coordinates for an RNA sequence."""
    length = len(sequence)
    
    # Add variation to parameters for different models
    np.random.seed(variation_idx * 42)
    variation = 0.9 + 0.2 * np.random.random()
    
    params = {
        'rise': A_FORM_PARAMS['rise'] * (0.9 + 0.2 * np.random.random()),
        'twist': np.deg2rad(A_FORM_PARAMS['twist'] * (0.9 + 0.2 * np.random.random())),
        'radius': A_FORM_PARAMS['radius'] * variation
    }
    
    coords = []
    for i in range(length):
        angle = i * params['twist']
        x = params['radius'] * np.cos(angle)
        y = params['radius'] * np.sin(angle)
        z = i * params['rise']
        coords.append([x, y, z])
    
    coords = np.array(coords)
    
    # Apply random rotation for variation
    if variation_idx > 0:
        rotation = Rotation.from_euler('xyz', 
            [np.random.uniform(-30, 30) for _ in range(3)], 
            degrees=True).as_matrix()
        coords = coords @ rotation
    
    # Center the structure
    coords = coords - np.mean(coords, axis=0)
    
    return coords

def create_submission(test_file, output_file='submission.csv'):
    """Create submission CSV from test sequences."""
    test_df = pd.read_csv(test_file)
    
    all_data = []
    
    for _, row in test_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        print(f"Processing {target_id} (len={len(sequence)})")
        
        # Generate 5 structure models
        models = [generate_helix_coords(sequence, i) for i in range(5)]
        
        # Build rows for each residue
        for i, nucleotide in enumerate(sequence):
            row_data = {
                'ID': f'{target_id}_{i+1}',
                'resname': nucleotide,
                'resid': i + 1
            }
            
            for model_idx, model_coords in enumerate(models, 1):
                row_data[f'x_{model_idx}'] = float(model_coords[i, 0])
                row_data[f'y_{model_idx}'] = float(model_coords[i, 1])
                row_data[f'z_{model_idx}'] = float(model_coords[i, 2])
            
            all_data.append(row_data)
    
    # Create DataFrame with correct column order
    submission_df = pd.DataFrame(all_data)
    column_order = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        column_order.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    submission_df = submission_df[column_order]
    submission_df.to_csv(output_file, index=False, float_format='%.3f')
    
    print(f"\nSubmission saved to {output_file}")
    print(f"Shape: {submission_df.shape}")
    
    return submission_df

if __name__ == '__main__':
    submission = create_submission('test_sequences.csv', 'submission.csv')
    print(submission.head(10))
