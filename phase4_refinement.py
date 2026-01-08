#!/usr/bin/env python3
"""
🧬 Phase 4: Structure Refinement & Energy Optimization
Stanford RNA 3D Folding Competition - Part 2

Strategy:
1. Secondary structure prediction integration
2. Base pairing constraint satisfaction
3. Energy-based structure refinement
4. Clash detection and resolution
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

KAGGLE_MODE = os.path.exists('/kaggle/input')

CONFIG = {
    'data_dir': '/kaggle/input/stanford-rna-3d-folding-2/' if KAGGLE_MODE else './data/',
    'min_c1_distance': 3.5,   # Minimum C1'-C1' distance (Å)
    'bp_distance': 10.0,      # Expected base pair C1'-C1' distance (Å)
    'bp_tolerance': 2.0,      # Base pair distance tolerance (Å)
    'stack_distance': 4.5,    # Expected stacking distance (Å)
    'max_iterations': 100,
    'refinement_lr': 0.01,
}

# Base pairing rules
BASE_PAIRS = {
    ('A', 'U'), ('U', 'A'),
    ('G', 'C'), ('C', 'G'),
    ('G', 'U'), ('U', 'G'),  # Wobble pairs
}

# A-form helix parameters
A_FORM_PARAMS = {
    'rise': 2.8,
    'twist': 32.7,
    'radius': 9.0
}

# =============================================================================
# Secondary Structure Prediction
# =============================================================================

class SecondaryStructurePredictor:
    """Predict RNA secondary structure using various methods."""
    
    def __init__(self):
        self.viennarna_available = self._check_viennarna()
        self.nussinov_available = True  # Always available (pure Python)
    
    def _check_viennarna(self):
        """Check if ViennaRNA is available."""
        try:
            import RNA
            return True
        except ImportError:
            return False
    
    def predict(self, sequence):
        """
        Predict secondary structure.
        
        Returns:
        --------
        str : Dot-bracket notation
        list : List of base pairs (i, j)
        """
        if self.viennarna_available:
            return self._viennarna_predict(sequence)
        else:
            return self._nussinov_predict(sequence)
    
    def _viennarna_predict(self, sequence):
        """Use ViennaRNA for prediction."""
        import RNA
        
        fc = RNA.fold_compound(sequence)
        structure, mfe = fc.mfe()
        base_pairs = self._parse_dotbracket(structure)
        
        return structure, base_pairs
    
    def _nussinov_predict(self, sequence):
        """Simple Nussinov algorithm for base pair prediction."""
        n = len(sequence)
        dp = np.zeros((n, n), dtype=int)
        
        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Case 1: i unpaired
                dp[i, j] = dp[i + 1, j]
                
                # Case 2: j unpaired
                dp[i, j] = max(dp[i, j], dp[i, j - 1])
                
                # Case 3: i-j paired
                if (sequence[i], sequence[j]) in BASE_PAIRS:
                    if i + 1 <= j - 1:
                        dp[i, j] = max(dp[i, j], dp[i + 1, j - 1] + 1)
                    else:
                        dp[i, j] = max(dp[i, j], 1)
                
                # Case 4: bifurcation
                for k in range(i + 1, j):
                    dp[i, j] = max(dp[i, j], dp[i, k] + dp[k + 1, j])
        
        # Traceback
        base_pairs = self._nussinov_traceback(dp, sequence, 0, n - 1)
        structure = self._pairs_to_dotbracket(base_pairs, n)
        
        return structure, base_pairs
    
    def _nussinov_traceback(self, dp, seq, i, j, pairs=None):
        """Traceback for Nussinov algorithm."""
        if pairs is None:
            pairs = []
        
        if i >= j:
            return pairs
        
        if dp[i, j] == dp[i + 1, j]:
            return self._nussinov_traceback(dp, seq, i + 1, j, pairs)
        elif dp[i, j] == dp[i, j - 1]:
            return self._nussinov_traceback(dp, seq, i, j - 1, pairs)
        elif (seq[i], seq[j]) in BASE_PAIRS:
            inner = dp[i + 1, j - 1] if i + 1 <= j - 1 else 0
            if dp[i, j] == inner + 1:
                pairs.append((i, j))
                return self._nussinov_traceback(dp, seq, i + 1, j - 1, pairs)
        
        # Bifurcation
        for k in range(i + 1, j):
            if dp[i, j] == dp[i, k] + dp[k + 1, j]:
                self._nussinov_traceback(dp, seq, i, k, pairs)
                self._nussinov_traceback(dp, seq, k + 1, j, pairs)
                return pairs
        
        return pairs
    
    def _parse_dotbracket(self, structure):
        """Parse dot-bracket notation to base pairs."""
        stack = []
        pairs = []
        
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        
        return pairs
    
    def _pairs_to_dotbracket(self, pairs, length):
        """Convert base pairs to dot-bracket notation."""
        structure = ['.'] * length
        for i, j in pairs:
            structure[i] = '('
            structure[j] = ')'
        return ''.join(structure)


# =============================================================================
# Energy Functions
# =============================================================================

def compute_clash_energy(coords, min_distance=3.5):
    """Compute steric clash penalty."""
    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)
    
    # Penalize distances below minimum
    clashes = distances < min_distance
    if np.any(clashes):
        violations = min_distance - distances[clashes]
        return np.sum(violations ** 2) * 100
    return 0.0


def compute_bond_energy(coords, expected_distance=5.5, tolerance=1.0):
    """Compute sequential bond length energy."""
    energy = 0.0
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(coords[i + 1] - coords[i])
        deviation = abs(dist - expected_distance)
        if deviation > tolerance:
            energy += (deviation - tolerance) ** 2
    return energy


def compute_basepair_energy(coords, base_pairs, expected_distance=10.0, tolerance=2.0):
    """Compute base pairing distance energy."""
    energy = 0.0
    for i, j in base_pairs:
        if i < len(coords) and j < len(coords):
            dist = np.linalg.norm(coords[i] - coords[j])
            deviation = abs(dist - expected_distance)
            if deviation > tolerance:
                energy += (deviation - tolerance) ** 2 * 10  # Higher weight for base pairs
    return energy


def compute_total_energy(coords, base_pairs=None):
    """Compute total energy of structure."""
    energy = 0.0
    
    # Clash energy
    energy += compute_clash_energy(coords)
    
    # Bond energy
    energy += compute_bond_energy(coords)
    
    # Base pair energy
    if base_pairs:
        energy += compute_basepair_energy(coords, base_pairs)
    
    return energy


# =============================================================================
# Structure Refinement
# =============================================================================

class StructureRefiner:
    """Refine RNA structures using energy minimization."""
    
    def __init__(self):
        self.ss_predictor = SecondaryStructurePredictor()
    
    def refine(self, coords, sequence, max_iterations=100, lr=0.01):
        """
        Refine structure coordinates.
        
        Parameters:
        -----------
        coords : np.ndarray
            (N, 3) array of initial coordinates
        sequence : str
            RNA sequence
        max_iterations : int
            Maximum optimization iterations
        lr : float
            Learning rate for gradient descent
        
        Returns:
        --------
        np.ndarray : Refined coordinates
        """
        # Predict secondary structure
        structure, base_pairs = self.ss_predictor.predict(sequence)
        
        # Define objective function
        def objective(x):
            c = x.reshape(-1, 3)
            return compute_total_energy(c, base_pairs)
        
        # Optimize
        x0 = coords.flatten()
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        refined_coords = result.x.reshape(-1, 3)
        
        # Report improvement
        initial_energy = compute_total_energy(coords, base_pairs)
        final_energy = result.fun
        
        return refined_coords, {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'improvement': (initial_energy - final_energy) / initial_energy if initial_energy > 0 else 0,
            'base_pairs': len(base_pairs),
            'structure': structure
        }
    
    def resolve_clashes(self, coords, min_distance=3.5, max_iterations=50):
        """Resolve steric clashes by pushing atoms apart."""
        coords = coords.copy()
        
        for iteration in range(max_iterations):
            distances = squareform(pdist(coords))
            np.fill_diagonal(distances, np.inf)
            
            # Find clashing pairs
            clashes = np.argwhere(distances < min_distance)
            
            if len(clashes) == 0:
                break
            
            # Push clashing atoms apart
            for i, j in clashes:
                if i < j:
                    vec = coords[j] - coords[i]
                    dist = np.linalg.norm(vec)
                    if dist < 0.1:
                        vec = np.random.randn(3)
                        dist = np.linalg.norm(vec)
                    
                    vec = vec / dist
                    push = (min_distance - dist) / 2 + 0.1
                    
                    coords[i] -= vec * push
                    coords[j] += vec * push
        
        return coords


# =============================================================================
# Secondary Structure Constraints
# =============================================================================

def apply_ss_constraints(coords, sequence, base_pairs, bp_distance=10.0):
    """Apply secondary structure constraints to coordinates."""
    coords = coords.copy()
    
    # Adjust base paired residues
    for i, j in base_pairs:
        if i < len(coords) and j < len(coords):
            current_dist = np.linalg.norm(coords[i] - coords[j])
            
            if abs(current_dist - bp_distance) > 2.0:
                # Move residues closer/further
                vec = coords[j] - coords[i]
                dist = np.linalg.norm(vec)
                if dist > 0.1:
                    vec = vec / dist
                    adjustment = (bp_distance - dist) / 2
                    coords[i] -= vec * adjustment * 0.3
                    coords[j] += vec * adjustment * 0.3
    
    return coords


# =============================================================================
# A-Form Helix Generation
# =============================================================================

def generate_aform_helix_coords(sequence, variation_idx=0):
    """Generate A-form helix C1' coordinates."""
    np.random.seed(variation_idx * 42)
    
    rise = A_FORM_PARAMS['rise'] * (0.9 + 0.2 * np.random.random())
    twist = np.deg2rad(A_FORM_PARAMS['twist'] * (0.9 + 0.2 * np.random.random()))
    radius = A_FORM_PARAMS['radius'] * (0.9 + 0.2 * np.random.random())
    
    coords = []
    for i in range(len(sequence)):
        angle = i * twist
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = i * rise
        coords.append([x, y, z])
    
    coords = np.array(coords)
    
    # Apply random rotation for variation
    if variation_idx > 0:
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', 
            [np.random.uniform(-30, 30) for _ in range(3)], 
            degrees=True).as_matrix()
        coords = coords @ rotation
    
    # Center
    coords = coords - np.mean(coords, axis=0)
    
    return coords


# =============================================================================
# Main Processing
# =============================================================================

def refine_predictions(predictions, sequences, targets):
    """Refine all predictions."""
    print("\n" + "=" * 60)
    print("🔧 STRUCTURE REFINEMENT")
    print("=" * 60)
    
    refiner = StructureRefiner()
    refined_predictions = {}
    
    for target, sequence in zip(targets, sequences):
        print(f"\n   Refining {target} (len={len(sequence)})")
        
        if target in predictions:
            target_preds = predictions[target]
        else:
            # Generate fallback
            target_preds = [generate_aform_helix_coords(sequence, i) for i in range(5)]
        
        refined = []
        for i, coords in enumerate(target_preds[:5]):
            if isinstance(coords, np.ndarray):
                # Resolve clashes
                coords = refiner.resolve_clashes(coords)
                
                # Refine with energy minimization (only for shorter sequences)
                if len(sequence) <= 200:
                    coords, stats = refiner.refine(coords, sequence, max_iterations=50)
                    if i == 0:  # Report only for first model
                        print(f"      Energy: {stats['initial_energy']:.1f} → {stats['final_energy']:.1f}")
                        print(f"      Base pairs: {stats['base_pairs']}")
                
                refined.append(coords)
            else:
                refined.append(generate_aform_helix_coords(sequence, i))
        
        # Fill to 5 models
        while len(refined) < 5:
            refined.append(generate_aform_helix_coords(sequence, len(refined)))
        
        refined_predictions[target] = refined[:5]
    
    return refined_predictions


def create_submission(refined_predictions, test_df, output_file='submission_phase4_refined.csv'):
    """Create submission from refined predictions."""
    print("\n" + "=" * 60)
    print("📤 CREATING SUBMISSION")
    print("=" * 60)
    
    output_labels = []
    
    for _, row in test_df.iterrows():
        target = row['target_id']
        sequence = row['sequence']
        
        predictions = refined_predictions.get(target, 
            [generate_aform_helix_coords(sequence, i) for i in range(5)])
        
        for i in range(len(sequence)):
            row_data = {
                'ID': f'{target}_{i+1}',
                'resname': sequence[i],
                'resid': i + 1,
            }
            
            for model_idx, coords in enumerate(predictions, 1):
                row_data[f'x_{model_idx}'] = float(coords[i, 0])
                row_data[f'y_{model_idx}'] = float(coords[i, 1])
                row_data[f'z_{model_idx}'] = float(coords[i, 2])
            
            output_labels.append(row_data)
    
    submission_df = pd.DataFrame(output_labels)
    
    # Ensure column order
    column_order = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        column_order.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    submission_df = submission_df[column_order]
    submission_df.to_csv(output_file, index=False, float_format='%.3f')
    
    print(f"\n✓ Submission saved to {output_file}")
    print(f"  Shape: {submission_df.shape}")
    
    return submission_df


# =============================================================================
# Main Execution
# =============================================================================

def run_phase4(previous_predictions=None):
    """Execute Phase 4: Structure Refinement."""
    print("\n" + "=" * 60)
    print("🧬 PHASE 4: STRUCTURE REFINEMENT & ENERGY OPTIMIZATION")
    print("   Stanford RNA 3D Folding Competition - Part 2")
    print("=" * 60)
    
    # Load data
    test_df = pd.read_csv(CONFIG['data_dir'] + 'test_sequences.csv')
    print(f"\n✓ Loaded {len(test_df)} test sequences")
    
    targets = test_df['target_id'].tolist()
    sequences = test_df['sequence'].tolist()
    
    # Generate initial predictions if not provided
    if previous_predictions is None:
        print("\n⚠️ No previous predictions provided, generating fallback structures")
        previous_predictions = {}
        for target, seq in zip(targets, sequences):
            previous_predictions[target] = [
                generate_aform_helix_coords(seq, i) for i in range(5)
            ]
    
    # Refine predictions
    refined_predictions = refine_predictions(previous_predictions, sequences, targets)
    
    # Create submission
    submission_df = create_submission(refined_predictions, test_df)
    
    # Validate
    sample_sub = pd.read_csv(CONFIG['data_dir'] + 'sample_submission.csv')
    print(f"\n🔍 Validation:")
    print(f"   ID match: {(submission_df['ID'] == sample_sub['ID']).all()}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PHASE 4 SUMMARY")
    print("=" * 60)
    print("""
    Completed:
    • Secondary structure prediction
    • Base pairing constraint application
    • Energy minimization
    • Steric clash resolution
    
    Next Steps (Phase 5):
    • Final ensemble combination
    • Model selection optimization
    • Submission pipeline finalization
    """)
    
    return submission_df, refined_predictions


if __name__ == '__main__':
    submission_df, refined = run_phase4()
