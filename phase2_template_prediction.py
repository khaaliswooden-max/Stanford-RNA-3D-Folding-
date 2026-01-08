#!/usr/bin/env python3
"""
🧬 Phase 2: Template-Based Structure Prediction
Stanford RNA 3D Folding Competition - Part 2

Strategy:
1. Use MMseqs2 to find homologous template structures from PDB
2. Extract C1' coordinates from matched templates
3. Handle sequence alignment and coordinate mapping
4. Apply temporal cutoff filtering
"""

import pandas as pd
import numpy as np
import os
import gzip
import csv
import subprocess
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

KAGGLE_MODE = os.path.exists('/kaggle/input')

CONFIG = {
    'data_dir': '/kaggle/input/stanford-rna-3d-folding-2/' if KAGGLE_MODE else './data/',
    'cif_dir': '/kaggle/input/stanford-rna-3d-folding-2/PDB_RNA' if KAGGLE_MODE else './data/PDB_RNA/',
    'mmseqs_path': '/kaggle/working/mmseqs/bin/mmseqs' if KAGGLE_MODE else 'mmseqs',
    'check_temporal_cutoff': True,
    'max_templates': 5,
    'null_value': 0.0,  # Use np.nan for debugging
    'min_alignment_coverage': 0.3,
    'max_evalue': 1e-3,
}

# A-form helix fallback parameters
A_FORM_PARAMS = {
    'rise': 2.8,
    'twist': 32.7,
    'radius': 9.0
}

# =============================================================================
# BioPython Imports (conditional)
# =============================================================================

try:
    from Bio import SeqIO, PDB, BiopythonWarning
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.Seq import Seq
    from Bio.PDB import MMCIFParser
    warnings.simplefilter('ignore', BiopythonWarning)
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("⚠️ BioPython not installed. Install with: pip install biopython")

# =============================================================================
# MMseqs2 Setup Functions
# =============================================================================

def setup_mmseqs():
    """Setup MMseqs2 for sequence search."""
    print("=" * 60)
    print("🔧 SETTING UP MMseqs2")
    print("=" * 60)
    
    if KAGGLE_MODE:
        # Copy MMseqs2 binary
        if not os.path.exists('/kaggle/working/mmseqs'):
            subprocess.run(['rsync', '-avL', '/kaggle/input/mmseqs2/mmseqs', '/kaggle/working/'], 
                         capture_output=True)
            subprocess.run(['chmod', '755', '/kaggle/working/mmseqs/bin/mmseqs'], 
                         capture_output=True)
        print("   ✓ MMseqs2 installed")
        
        # Create sequence database
        db_path = '/kaggle/working/pdb_seqres_NA'
        if not os.path.exists(db_path + '.dbtype'):
            fasta_path = CONFIG['cif_dir'] + '/pdb_seqres_NA.fasta'
            subprocess.run([CONFIG['mmseqs_path'], 'createdb', fasta_path, db_path, 
                          '--dbtype', '2'], capture_output=True)
            print("   ✓ PDB sequence database created")
    else:
        print("   ⚠️ Local mode - ensure MMseqs2 is in PATH")


def convert_to_fasta(test_df, output_file='test_sequences.fasta'):
    """Convert test sequences to FASTA format."""
    with open(output_file, 'w') as f:
        for _, row in test_df.iterrows():
            f.write(f">{row['target_id']}\n{row['sequence']}\n")
    print(f"   ✓ Converted to FASTA: {output_file}")


def run_mmseqs_search(fasta_file, output_file='testResult.txt'):
    """Run MMseqs2 easy-search."""
    print("\n" + "=" * 60)
    print("🔍 RUNNING MMseqs2 SEARCH")
    print("=" * 60)
    
    if KAGGLE_MODE:
        db_path = '/kaggle/working/pdb_seqres_NA'
        cmd = [
            CONFIG['mmseqs_path'], 'easy-search',
            fasta_file, db_path, output_file, 'tmp',
            '--search-type', '3',
            '--format-output', 'query,target,evalue,qstart,qend,tstart,tend,qaln,taln'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ MMseqs2 search completed")
        else:
            print(f"   ⚠️ MMseqs2 error: {result.stderr}")
    else:
        print("   ⚠️ Local mode - using existing results file")

# =============================================================================
# CIF Parsing Functions
# =============================================================================

def clean_res_name(res_name):
    """Clean residue name to standard nucleotide."""
    if res_name in ['A', 'C', 'G', 'U']:
        return res_name
    return 'X'


def is_before_or_on(date_str1, date_str2):
    """Check if date_str1 is before or on date_str2."""
    try:
        d1 = datetime.strptime(str(date_str1)[:10], '%Y-%m-%d')
        d2 = datetime.strptime(str(date_str2)[:10], '%Y-%m-%d')
        return d1 <= d2
    except:
        return False


def read_release_dates(csv_path):
    """Read PDB release dates from CSV."""
    release_dates = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            release_dates[row['pdb_id'].upper()] = row['release_date']
    return release_dates


def extract_title_release_date(cif_path):
    """Extract title and release date from CIF file."""
    if not BIOPYTHON_AVAILABLE:
        return None, None
        
    try:
        if cif_path.endswith('.gz'):
            with gzip.open(cif_path, 'rt') as cif_file:
                mmcif_dict = MMCIF2Dict(cif_file)
        else:
            mmcif_dict = MMCIF2Dict(cif_path)
        
        # Get title
        title_fields = ['_struct.title', '_entry.title', '_struct_keywords.pdbx_keywords']
        pdb_title = None
        for field in title_fields:
            if field in mmcif_dict:
                pdb_title = mmcif_dict[field]
                if isinstance(pdb_title, list):
                    pdb_title = ' '.join(pdb_title)
                break
        
        # Get release date
        date_fields = [
            '_pdbx_database_status.initial_release_date',
            '_pdbx_database_status.recvd_initial_deposition_date',
            '_database_PDB_rev.date'
        ]
        release_date = None
        for field in date_fields:
            if field in mmcif_dict:
                release_date = mmcif_dict[field]
                if isinstance(release_date, list):
                    release_date = release_date[0]
                break
        
        return pdb_title, release_date
    except Exception as e:
        return None, None


def extract_rna_sequence(cif_path, chain_id):
    """Extract RNA sequence from CIF file."""
    if not BIOPYTHON_AVAILABLE:
        return '', '', []
        
    try:
        if cif_path.endswith('.gz'):
            with gzip.open(cif_path, 'rt') as cif_file:
                mmcif_dict = MMCIF2Dict(cif_file)
        else:
            mmcif_dict = MMCIF2Dict(cif_path)
        
        strand_id = mmcif_dict.get('_pdbx_poly_seq_scheme.pdb_strand_id', [])
        mon_id = mmcif_dict.get('_pdbx_poly_seq_scheme.mon_id', [])
        pdb_mon_id = mmcif_dict.get('_pdbx_poly_seq_scheme.pdb_mon_id', [])
        pdb_seq_num = mmcif_dict.get('_pdbx_poly_seq_scheme.pdb_seq_num', [])
        
        full_sequence = ''
        pdb_chain_sequence = ''
        pdb_chain_seq_nums = []
        
        for (strand, mon, pdb_mon, pdb_num) in zip(strand_id, mon_id, pdb_mon_id, pdb_seq_num):
            if strand == chain_id:
                full_sequence += clean_res_name(mon)
                pdb_chain_sequence += clean_res_name(pdb_mon)
                pdb_chain_seq_nums.append(pdb_num)
        
        return full_sequence, pdb_chain_sequence, pdb_chain_seq_nums
    except Exception as e:
        return '', '', []


def get_c1prime_labels(cif_path, chain_id, alignment, chain_seq_nums):
    """Extract C1' coordinates for an RNA chain based on alignment."""
    if not BIOPYTHON_AVAILABLE:
        return []
        
    try:
        if cif_path.endswith('.gz'):
            parser = MMCIFParser(QUIET=True)
            with gzip.open(cif_path, 'rt') as handle:
                structure = parser.get_structure('rna', handle)
        else:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('rna', cif_path)
        
        # Get C1' coordinates for the chain
        c1_coords = {}
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        res_id = residue.id[1]
                        res_name = residue.resname.strip()
                        if len(res_name) == 1 or res_name in ['A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT']:
                            for atom in residue:
                                if atom.name == "C1'":
                                    c1_coords[str(res_id)] = {
                                        'coords': atom.coord,
                                        'resname': clean_res_name(res_name[-1] if len(res_name) > 1 else res_name)
                                    }
                                    break
            break  # Use first model only
        
        # Map alignment to coordinates
        query_seq = alignment[0]
        template_seq = alignment[1]
        
        result = []
        seq_idx = 0
        template_idx = 0
        
        for q_char, t_char in zip(query_seq, template_seq):
            if q_char != '-':
                seq_idx += 1
                if t_char != '-' and t_char != 'X':
                    template_idx += 1
                    if template_idx <= len(chain_seq_nums):
                        pdb_seq_num = chain_seq_nums[template_idx - 1]
                        if pdb_seq_num in c1_coords:
                            coord_data = c1_coords[pdb_seq_num]
                            result.append((
                                coord_data['resname'],
                                seq_idx,
                                coord_data['coords'][0],
                                coord_data['coords'][1],
                                coord_data['coords'][2],
                                pdb_seq_num
                            ))
                        else:
                            result.append((q_char, seq_idx, CONFIG['null_value'], 
                                         CONFIG['null_value'], CONFIG['null_value'], None))
                    else:
                        result.append((q_char, seq_idx, CONFIG['null_value'], 
                                     CONFIG['null_value'], CONFIG['null_value'], None))
                else:
                    if t_char != '-':
                        template_idx += 1
                    result.append((q_char, seq_idx, CONFIG['null_value'], 
                                 CONFIG['null_value'], CONFIG['null_value'], None))
            else:
                if t_char != '-':
                    template_idx += 1
        
        return result
    except Exception as e:
        return []

# =============================================================================
# Fallback A-Form Helix
# =============================================================================

def generate_aform_helix(sequence, variation_idx=0):
    """Generate A-form helix coordinates as fallback."""
    rise = A_FORM_PARAMS['rise'] * (0.9 + 0.2 * (variation_idx * 0.1))
    twist = np.deg2rad(A_FORM_PARAMS['twist'] * (0.95 + 0.1 * variation_idx))
    radius = A_FORM_PARAMS['radius'] * (0.9 + 0.2 * (variation_idx * 0.05))
    
    coords = []
    for i, nuc in enumerate(sequence):
        angle = i * twist
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = i * rise
        coords.append((nuc, i + 1, x, y, z, None))
    
    return coords

# =============================================================================
# Template Matching Pipeline
# =============================================================================

def parse_mmseqs_results(results_file):
    """Parse MMseqs2 results file."""
    alignments = defaultdict(list)
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 9:
                    query, target, evalue, qstart, qend, tstart, tend, qaln, taln = parts[:9]
                    alignments[query].append({
                        'target': target,
                        'evalue': float(evalue),
                        'qstart': int(qstart),
                        'qend': int(qend),
                        'tstart': int(tstart),
                        'tend': int(tend),
                        'qaln': qaln,
                        'taln': taln
                    })
    
    return alignments


def find_templates(target_id, sequence, temporal_cutoff, alignments, release_dates):
    """Find valid templates for a target sequence."""
    templates = []
    
    if target_id not in alignments:
        return templates
    
    for aln in alignments[target_id]:
        if aln['qend'] < aln['qstart']:
            continue  # Skip reverse complement
        
        if aln['evalue'] > CONFIG['max_evalue']:
            continue
        
        # Parse PDB ID and chain
        target_parts = aln['target'].split('_')
        if len(target_parts) < 2:
            continue
            
        pdb_id = target_parts[0]
        chain_id = target_parts[1]
        
        # Find CIF file
        cif_path = os.path.join(CONFIG['cif_dir'], f'{pdb_id.lower()}.cif')
        if not os.path.isfile(cif_path):
            cif_path = os.path.join(CONFIG['cif_dir'], f'{pdb_id.lower()}.cif.gz')
            if not os.path.isfile(cif_path):
                continue
        
        # Check temporal cutoff
        if CONFIG['check_temporal_cutoff'] and pdb_id.upper() in release_dates:
            release_date = release_dates[pdb_id.upper()]
            if is_before_or_on(temporal_cutoff, release_date):
                continue
        
        try:
            # Get template coordinates
            chain_full_seq, chain_seq, chain_seq_nums = extract_rna_sequence(cif_path, chain_id)
            
            if not chain_seq_nums:
                continue
            
            # Build alignment
            qstart, qend = aln['qstart'], aln['qend']
            tstart, tend = aln['tstart'], aln['tend']
            
            alignment = [
                sequence[:(qstart-1)] + '-'*(tstart-1) + aln['qaln'] + sequence[qend:],
                '-'*(qstart-1) + 'X'*(tstart-1) + aln['taln'] + '-'*(len(sequence)-qend)
            ]
            
            c1prime_data = get_c1prime_labels(cif_path, chain_id, alignment, chain_seq_nums)
            
            if len(c1prime_data) == len(sequence):
                # Calculate coverage
                valid_coords = sum(1 for d in c1prime_data if d[2] != CONFIG['null_value'])
                coverage = valid_coords / len(sequence)
                
                if coverage >= CONFIG['min_alignment_coverage']:
                    templates.append({
                        'pdb_id': pdb_id,
                        'chain_id': chain_id,
                        'coverage': coverage,
                        'evalue': aln['evalue'],
                        'coords': c1prime_data
                    })
        except Exception as e:
            continue
        
        if len(templates) >= CONFIG['max_templates']:
            break
    
    # Sort by coverage
    templates.sort(key=lambda x: x['coverage'], reverse=True)
    
    return templates[:CONFIG['max_templates']]

# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_all_targets(test_df, alignments, release_dates):
    """Process all targets and generate predictions."""
    print("\n" + "=" * 60)
    print("🔬 PROCESSING TARGETS")
    print("=" * 60)
    
    output_labels = []
    targets = test_df['target_id'].tolist()
    sequences = test_df['sequence'].tolist()
    temporal_cutoffs = test_df['temporal_cutoff'].tolist() if 'temporal_cutoff' in test_df.columns else [None] * len(targets)
    
    stats = {'template_hits': 0, 'fallback_used': 0}
    
    for count, (target, sequence, temporal_cutoff) in enumerate(zip(targets, sequences, temporal_cutoffs)):
        print(f"\n[{count+1}/{len(targets)}] Processing {target} (len={len(sequence)})")
        
        # Find templates
        templates = find_templates(target, sequence, temporal_cutoff, alignments, release_dates)
        
        if templates:
            stats['template_hits'] += 1
            for t in templates:
                print(f"   ✓ Template: {t['pdb_id']}_{t['chain_id']} (coverage: {t['coverage']:.2%})")
        
        # Fill remaining slots with A-form helix
        template_coords = [t['coords'] for t in templates]
        while len(template_coords) < CONFIG['max_templates']:
            variation_idx = len(template_coords)
            template_coords.append(generate_aform_helix(sequence, variation_idx))
            if variation_idx == len(templates):  # First fallback
                stats['fallback_used'] += 1
                print(f"   → Added A-form helix fallback")
        
        # Build output rows
        for i in range(len(sequence)):
            output_label = {
                'ID': f'{target}_{i+1}',
                'resname': sequence[i],
                'resid': i + 1,
            }
            
            for n in range(CONFIG['max_templates']):
                res, resid, x, y, z, pdb_seqnum = template_coords[n][i]
                output_label[f'x_{n+1}'] = x
                output_label[f'y_{n+1}'] = y
                output_label[f'z_{n+1}'] = z
            
            output_labels.append(output_label)
    
    print(f"\n📊 Statistics:")
    print(f"   Targets with templates: {stats['template_hits']}/{len(targets)}")
    print(f"   Targets using fallback: {stats['fallback_used']}/{len(targets)}")
    
    return output_labels


def create_submission(output_labels, output_file='submission_phase2_template.csv'):
    """Create submission DataFrame."""
    print("\n" + "=" * 60)
    print("📤 CREATING SUBMISSION")
    print("=" * 60)
    
    submission_df = pd.DataFrame(output_labels)
    
    # Ensure column order
    column_order = ['ID', 'resname', 'resid']
    for i in range(1, CONFIG['max_templates'] + 1):
        column_order.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    submission_df = submission_df[column_order]
    submission_df.to_csv(output_file, index=False, float_format='%.3f')
    
    print(f"\n✓ Submission saved to {output_file}")
    print(f"  Shape: {submission_df.shape}")
    
    return submission_df

# =============================================================================
# Main Execution
# =============================================================================

def run_phase2():
    """Execute Phase 2: Template-Based Prediction."""
    print("\n" + "=" * 60)
    print("🧬 PHASE 2: TEMPLATE-BASED STRUCTURE PREDICTION")
    print("   Stanford RNA 3D Folding Competition - Part 2")
    print("=" * 60)
    
    # Load data
    test_df = pd.read_csv(CONFIG['data_dir'] + 'test_sequences.csv')
    print(f"\n✓ Loaded {len(test_df)} test sequences")
    
    # Setup MMseqs2
    setup_mmseqs()
    
    # Convert to FASTA and run search
    convert_to_fasta(test_df, 'test_sequences.fasta')
    run_mmseqs_search('test_sequences.fasta', 'testResult.txt')
    
    # Parse results
    alignments = parse_mmseqs_results('testResult.txt')
    print(f"\n✓ Parsed alignments for {len(alignments)} targets")
    
    # Read release dates
    release_dates = read_release_dates(CONFIG['cif_dir'] + '/pdb_release_dates_NA.csv')
    print(f"✓ Loaded {len(release_dates)} PDB release dates")
    
    # Process all targets
    output_labels = process_all_targets(test_df, alignments, release_dates)
    
    # Create submission
    submission_df = create_submission(output_labels)
    
    # Validate
    sample_sub = pd.read_csv(CONFIG['data_dir'] + 'sample_submission.csv')
    print(f"\n🔍 Validation:")
    print(f"   ID match: {(submission_df['ID'] == sample_sub['ID']).all()}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PHASE 2 SUMMARY")
    print("=" * 60)
    print("""
    Completed:
    • MMseqs2 sequence similarity search
    • Template coordinate extraction from PDB structures
    • Temporal cutoff filtering
    • A-form helix fallback for missing coverage
    
    Next Steps (Phase 3):
    • Integrate deep learning methods (RhoFold/ESMFold)
    • Improve structure prediction accuracy
    • Handle sequences without template matches
    """)
    
    return submission_df


if __name__ == '__main__':
    submission_df = run_phase2()
