#!/usr/bin/env python3
"""
🧬 Phase 1: Data Exploration & Baseline Setup
Stanford RNA 3D Folding Competition - Part 2

Objective: Predict C1' atom 3D coordinates for RNA sequences

Phase Goals:
1. Explore competition data structure
2. Understand submission format
3. Analyze sequence distributions
4. Create A-form helix baseline
5. Validate submission pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# File paths - Update for local vs Kaggle
KAGGLE_MODE = os.path.exists('/kaggle/input')
DATA_DIR = '/kaggle/input/stanford-rna-3d-folding-2/' if KAGGLE_MODE else './data/'

# A-form RNA helix parameters (Angstroms)
A_FORM_PARAMS = {
    'rise': 2.8,      # Rise per residue along helix axis
    'twist': 32.7,    # Twist angle per residue (degrees)
    'radius': 9.0     # Radius of helix
}

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_competition_data():
    """Load all competition data files."""
    print("=" * 60)
    print("📊 LOADING COMPETITION DATA")
    print("=" * 60)
    
    test_sequences = pd.read_csv(DATA_DIR + 'test_sequences.csv')
    sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
    
    print(f"\n✓ Test sequences: {len(test_sequences)} RNA molecules")
    print(f"✓ Sample submission: {len(sample_submission)} residues")
    print(f"\n🔬 Test sequences columns: {list(test_sequences.columns)}")
    print(f"🔬 Sample submission columns: {list(sample_submission.columns)}")
    
    return test_sequences, sample_submission

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_sequences(test_sequences):
    """Analyze sequence statistics and composition."""
    print("\n" + "=" * 60)
    print("🧬 SEQUENCE ANALYSIS")
    print("=" * 60)
    
    # Calculate sequence lengths
    test_sequences['seq_length'] = test_sequences['sequence'].apply(len)
    
    print("\n📊 Sequence Length Statistics:")
    print(f"   Min: {test_sequences['seq_length'].min()}")
    print(f"   Max: {test_sequences['seq_length'].max()}")
    print(f"   Mean: {test_sequences['seq_length'].mean():.1f}")
    print(f"   Median: {test_sequences['seq_length'].median()}")
    print(f"   Total residues: {test_sequences['seq_length'].sum()}")
    
    # Nucleotide composition
    all_sequences = ''.join(test_sequences['sequence'])
    nuc_counts = Counter(all_sequences)
    
    print("\n🧬 Nucleotide Composition:")
    for nuc in ['A', 'U', 'G', 'C']:
        count = nuc_counts.get(nuc, 0)
        pct = 100 * count / len(all_sequences)
        print(f"   {nuc}: {count:,} ({pct:.2f}%)")
    
    # Check for non-standard nucleotides
    non_standard = {k: v for k, v in nuc_counts.items() if k not in ['A', 'U', 'G', 'C']}
    if non_standard:
        print(f"\n⚠️ Non-standard nucleotides found: {non_standard}")
    
    return test_sequences, nuc_counts


def analyze_submission_format(sample_submission, test_sequences):
    """Analyze the required submission format."""
    print("\n" + "=" * 60)
    print("📋 SUBMISSION FORMAT ANALYSIS")
    print("=" * 60)
    
    print(f"\n   Total rows: {len(sample_submission):,}")
    print(f"   Columns: {list(sample_submission.columns)}")
    
    # Extract target IDs
    sample_submission['target_id'] = sample_submission['ID'].apply(
        lambda x: '_'.join(x.split('_')[:-1])
    )
    unique_targets = sample_submission['target_id'].nunique()
    print(f"\n   Unique targets: {unique_targets}")
    
    # Verify coordinate columns
    coord_cols = [c for c in sample_submission.columns if c.startswith(('x_', 'y_', 'z_'))]
    print(f"\n   Coordinate columns: {coord_cols}")
    print("   → 5 structure models required (x/y/z for models 1-5)")
    
    return sample_submission


def analyze_temporal_cutoffs(test_sequences):
    """Analyze temporal cutoff dates."""
    print("\n" + "=" * 60)
    print("📅 TEMPORAL CUTOFF ANALYSIS")
    print("=" * 60)
    
    if 'temporal_cutoff' in test_sequences.columns:
        test_sequences['cutoff_date'] = pd.to_datetime(test_sequences['temporal_cutoff'])
        
        print(f"\n   Earliest: {test_sequences['cutoff_date'].min()}")
        print(f"   Latest: {test_sequences['cutoff_date'].max()}")
        print(f"\n   Cutoff distribution:")
        print(test_sequences['temporal_cutoff'].value_counts().to_string())
    else:
        print("\n   ⚠️ No temporal_cutoff column found")

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_sequence_distribution(test_sequences):
    """Plot sequence length distribution."""
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(test_sequences['seq_length'], bins=50, 
                 color='#00CED1', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of RNA Sequence Lengths', 
                      fontsize=14, fontweight='bold')
    axes[0].axvline(test_sequences['seq_length'].median(), 
                    color='#FF6B6B', linestyle='--', 
                    label=f"Median: {test_sequences['seq_length'].median():.0f}")
    axes[0].legend()
    
    # Box plot
    bp = axes[1].boxplot(test_sequences['seq_length'], vert=True, 
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('#00CED1')
    bp['boxes'][0].set_alpha(0.7)
    axes[1].set_ylabel('Sequence Length', fontsize=12)
    axes[1].set_title('Sequence Length Box Plot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase1_sequence_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: phase1_sequence_distribution.png")


def plot_nucleotide_composition(nuc_counts):
    """Plot nucleotide composition."""
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nucleotides = ['A', 'U', 'G', 'C']
    counts = [nuc_counts.get(n, 0) for n in nucleotides]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax.bar(nucleotides, counts, color=colors, edgecolor='white', linewidth=2)
    ax.set_xlabel('Nucleotide', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Nucleotide Distribution in Test Set', 
                 fontsize=14, fontweight='bold')
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase1_nucleotide_composition.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: phase1_nucleotide_composition.png")

# =============================================================================
# A-Form Helix Baseline
# =============================================================================

def generate_aform_helix(sequence, variation_idx=0):
    """
    Generate A-form helix C1' coordinates for RNA sequence.
    
    Parameters:
    -----------
    sequence : str
        RNA sequence (A, U, G, C)
    variation_idx : int
        Index for parameter variation (0-4 for 5 models)
    
    Returns:
    --------
    numpy.ndarray : (N, 3) array of C1' coordinates
    """
    length = len(sequence)
    
    # Add controlled variation for different models
    np.random.seed(variation_idx * 42)
    
    params = {
        'rise': A_FORM_PARAMS['rise'] * (0.9 + 0.2 * np.random.random()),
        'twist': np.deg2rad(A_FORM_PARAMS['twist'] * (0.9 + 0.2 * np.random.random())),
        'radius': A_FORM_PARAMS['radius'] * (0.9 + 0.2 * np.random.random())
    }
    
    coords = []
    for i in range(length):
        angle = i * params['twist']
        x = params['radius'] * np.cos(angle)
        y = params['radius'] * np.sin(angle)
        z = i * params['rise']
        coords.append([x, y, z])
    
    coords = np.array(coords)
    
    # Apply random rotation for model variation
    if variation_idx > 0:
        rotation = Rotation.from_euler('xyz', 
            [np.random.uniform(-30, 30) for _ in range(3)], 
            degrees=True).as_matrix()
        coords = coords @ rotation
    
    # Center the structure
    coords = coords - np.mean(coords, axis=0)
    
    return coords


def visualize_aform_helix():
    """Visualize A-form helix structure."""
    plt.style.use('dark_background')
    
    # Generate example structure
    example_seq = "AUGCAUGCAUGCAUGCAUGC"
    coords = generate_aform_helix(example_seq, 0)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by position
    colors = plt.cm.viridis(np.linspace(0, 1, len(coords)))
    
    # Plot backbone
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
            'w-', alpha=0.5, linewidth=1)
    
    # Plot residues
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c=colors, s=100, alpha=0.9)
    
    ax.set_xlabel('X (Å)', fontsize=11)
    ax.set_ylabel('Y (Å)', fontsize=11)
    ax.set_zlabel('Z (Å)', fontsize=11)
    ax.set_title('A-Form Helix Baseline Structure', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase1_aform_helix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: phase1_aform_helix.png")

# =============================================================================
# Submission Generation
# =============================================================================

def create_baseline_submission(test_df, sample_sub, output_file='submission_phase1_baseline.csv'):
    """
    Create baseline submission using A-form helix coordinates.
    """
    print("\n" + "=" * 60)
    print("📤 GENERATING BASELINE SUBMISSION")
    print("=" * 60)
    
    # Build sequence lookup
    seq_lookup = dict(zip(test_df['target_id'], test_df['sequence']))
    
    all_data = []
    current_target = None
    current_models = None
    
    for idx, row in sample_sub.iterrows():
        # Parse ID
        id_parts = row['ID'].rsplit('_', 1)
        target_id = id_parts[0]
        resid = int(id_parts[1])
        
        # Generate models for new target
        if target_id != current_target:
            current_target = target_id
            sequence = seq_lookup[target_id]
            current_models = [generate_aform_helix(sequence, i) for i in range(5)]
            print(f"   Processing {target_id} (len={len(sequence)})")
        
        # Get coordinates
        i = resid - 1
        row_data = {
            'ID': row['ID'],
            'resname': row['resname'],
            'resid': row['resid']
        }
        
        for model_idx, model_coords in enumerate(current_models, 1):
            row_data[f'x_{model_idx}'] = float(model_coords[i, 0])
            row_data[f'y_{model_idx}'] = float(model_coords[i, 1])
            row_data[f'z_{model_idx}'] = float(model_coords[i, 2])
        
        all_data.append(row_data)
    
    # Create DataFrame
    submission_df = pd.DataFrame(all_data)
    
    # Ensure column order
    column_order = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        column_order.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    submission_df = submission_df[column_order]
    submission_df.to_csv(output_file, index=False, float_format='%.3f')
    
    print(f"\n✓ Submission saved to {output_file}")
    print(f"  Shape: {submission_df.shape}")
    
    return submission_df


def validate_submission(submission_df, sample_submission):
    """Validate submission format."""
    print("\n" + "=" * 60)
    print("🔍 SUBMISSION VALIDATION")
    print("=" * 60)
    
    print(f"\n   ID match: {(submission_df['ID'] == sample_submission['ID']).all()}")
    print(f"   resname match: {(submission_df['resname'] == sample_submission['resname']).all()}")
    print(f"   resid match: {(submission_df['resid'] == sample_submission['resid']).all()}")
    
    # Check for NaN values
    nan_count = submission_df.isna().sum().sum()
    print(f"   NaN values: {nan_count}")
    
    if nan_count == 0:
        print("\n   ✓ Submission is valid!")
    else:
        print("\n   ⚠️ Submission contains NaN values - may fail validation")

# =============================================================================
# Main Execution
# =============================================================================

def run_phase1():
    """Execute Phase 1: Data Exploration & Baseline."""
    print("\n" + "=" * 60)
    print("🧬 PHASE 1: DATA EXPLORATION & BASELINE SETUP")
    print("   Stanford RNA 3D Folding Competition - Part 2")
    print("=" * 60)
    
    # Load data
    test_sequences, sample_submission = load_competition_data()
    
    # Analyze data
    test_sequences, nuc_counts = analyze_sequences(test_sequences)
    sample_submission = analyze_submission_format(sample_submission, test_sequences)
    analyze_temporal_cutoffs(test_sequences)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        plot_sequence_distribution(test_sequences)
        plot_nucleotide_composition(nuc_counts)
        visualize_aform_helix()
    except Exception as e:
        print(f"   ⚠️ Visualization error (may be headless): {e}")
    
    # Generate baseline submission
    submission_df = create_baseline_submission(test_sequences, sample_submission)
    
    # Validate submission
    validate_submission(submission_df, sample_submission)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PHASE 1 SUMMARY")
    print("=" * 60)
    print("""
    Key Findings:
    • Test set contains multiple RNA sequences of varying lengths
    • Submission requires 5 structure models per sequence
    • Each model provides C1' atom (x, y, z) coordinates
    • Temporal cutoffs must be respected for template-based methods
    
    Baseline Performance:
    • A-form helix provides a simple geometric baseline
    • Expected to have low accuracy but valid submission format
    
    Next Steps (Phase 2):
    • Implement template-based structure prediction using MMseqs2
    • Extract coordinates from PDB structures
    • Handle alignment and coordinate mapping
    """)
    
    return test_sequences, sample_submission, submission_df


if __name__ == '__main__':
    test_sequences, sample_submission, submission_df = run_phase1()
