# src/preprocess.py

import os
import warnings
import torch
import numpy as np
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch_geometric.data import Data
from tqdm import tqdm

# Import from our own project files
from src.config import (
    PDB_DIR, PROCESSED_DIR, ANTIBODY_CHAIN_IDS, EPITOPE_DISTANCE_THRESHOLD,
    GRAPH_EDGE_DISTANCE_THRESHOLD, NODE_FEATURE_DIM
)
from src.utils import (
    AMINO_ACIDS, AA_TO_INDEX, KYTE_DOOLITTLE_HYDROPHOBICITY,
    get_alpha_carbon, get_min_distance
)

# Suppress warnings from Bio.PDB about discontinuous chains, etc.
warnings.simplefilter('ignore', PDBConstructionWarning)


def get_protein_chains(structure):
    """Separates antigen and antibody chains from a PDB structure."""
    antigen_chains = []
    antibody_chains = []
    for model in structure:
        for chain in model:
            if chain.id in ANTIBODY_CHAIN_IDS:
                antibody_chains.append(chain)
            else:
                antigen_chains.append(chain)
    if not antigen_chains:
        raise ValueError("No antigen chains found in the structure.")
    if not antibody_chains:
        raise ValueError("No antibody chains found in the structure.")
    # For simplicity, we'll work with the first antigen chain found.
    # This can be extended to handle multi-chain antigens.
    return antigen_chains[0], antibody_chains

def calculate_features(residue, dssp_results):
    """Calculates a feature vector for a single residue."""
    # --- Feature 1: One-hot encoded amino acid type (20 dims) ---
    aa_type = residue.get_resname()
    if aa_type not in AA_TO_INDEX:
        return None  # Skip non-standard amino acids
    one_hot_aa = np.zeros(len(AMINO_ACIDS))
    one_hot_aa[AA_TO_INDEX[aa_type]] = 1

    # --- Feature 2 & 3: Secondary structure (SS) and Solvent Accessibility (SASA) ---
    res_id = residue.get_id()
    chain_id = residue.get_parent().id
    dssp_key = (chain_id, res_id)

    if dssp_key not in dssp_results:
        # Sometimes DSSP fails on a residue, we'll have to skip it
        return None

    dssp_info = dssp_results[dssp_key]
    ss = dssp_info[2]  # H, E, or G, C, S, T, B, I, -
    sasa = dssp_info[3] # Solvent Accessible Surface Area
    rsa = dssp_info[4]  # Relative Solvent Accessibility

    # One-hot encode secondary structure (3 dims: Helix, Strand, Coil)
    ss_one_hot = np.zeros(3)
    if ss in ['H', 'G', 'I']: # Helix types
        ss_one_hot[0] = 1
    elif ss in ['E', 'B']: # Strand types
        ss_one_hot[1] = 1
    else: # Coil and others
        ss_one_hot[2] = 1

    # --- Feature 4: Hydrophobicity (1 dim) ---
    hydrophobicity = KYTE_DOOLITTLE_HYDROPHOBICITY.get(aa_type, 0.0)

    # --- Concatenate all features ---
    # 20 (AA) + 1 (SASA) + 1 (RSA) + 3 (SS) + 1 (Hydrophobicity) = 26 dims
    # This must match NODE_FEATURE_DIM in config.py
    feature_vector = np.concatenate([
        one_hot_aa,
        [sasa],
        [rsa],
        ss_one_hot,
        [hydrophobicity]
    ])

    return feature_vector

def process_pdb(pdb_id):
    """
    Processes a single PDB file into a PyTorch Geometric Data object.

    Args:
        pdb_id (str): The PDB identifier (e.g., '1ADQ').

    Returns:
        torch_geometric.data.Data: A graph data object or None if processing fails.
    """
    parser = PDBParser(QUIET=True)
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"File not found: {pdb_path}")
        return None

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        antigen_chain, antibody_chains = get_protein_chains(structure)
        model = structure[0]
        dssp = DSSP(model, pdb_path)
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None

    node_features = []
    labels = []
    valid_residues = []

    # --- Step 1: Extract node features and labels for each residue ---
    for residue in antigen_chain.get_residues():
        # Skip non-standard residues or residues with missing atoms
        if residue.get_resname() not in AMINO_ACIDS or not get_alpha_carbon(residue):
            continue

        # Calculate feature vector
        features = calculate_features(residue, dssp)
        if features is None:
            continue
        
        # Calculate label (is it an epitope residue?)
        min_dist = float('inf')
        for ab_chain in antibody_chains:
            dist = get_min_distance(residue, ab_chain)
            if dist < min_dist:
                min_dist = dist
        
        label = 1 if min_dist < EPITOPE_DISTANCE_THRESHOLD else 0
        
        node_features.append(features)
        labels.append(label)
        valid_residues.append(residue)

    if not valid_residues:
        print(f"No valid residues found for antigen in {pdb_id}")
        return None

    # --- Step 2: Construct graph edges based on C-alpha distance ---
    edge_list = []
    num_nodes = len(valid_residues)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            res_i = valid_residues[i]
            res_j = valid_residues[j]
            ca_i = get_alpha_carbon(res_i)
            ca_j = get_alpha_carbon(res_j)
            dist = ca_i - ca_j
            if dist < GRAPH_EDGE_DISTANCE_THRESHOLD:
                edge_list.append([i, j])
                edge_list.append([j, i]) # Add edge in both directions for undirected graph

    # --- Step 3: Create PyTorch Geometric Data object ---
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.tensor(np.array(labels), dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Final check on feature dimension
    if x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(
            f"Feature dimension mismatch in {pdb_id}. "
            f"Expected {NODE_FEATURE_DIM}, got {x.shape[1]}"
        )

    data = Data(x=x, edge_index=edge_index, y=y, pdb_id=pdb_id)
    return data


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pdb_files = [f.split('.')[0] for f in os.listdir(PDB_DIR) if f.endswith('.pdb')]

    print(f"Found {len(pdb_files)} PDB files to process.")
    
    for pdb_id in tqdm(pdb_files, desc="Processing PDBs"):
        output_path = os.path.join(PROCESSED_DIR, f"{pdb_id}.pt")
        if os.path.exists(output_path):
            print(f"Skipping {pdb_id}, already processed.")
            continue

        graph_data = process_pdb(pdb_id)
        if graph_data:
            torch.save(graph_data, output_path)
            # print(f"Successfully processed and saved {pdb_id}.pt")

    print("\nPreprocessing complete.")
    print(f"Processed data saved to: {PROCESSED_DIR}")