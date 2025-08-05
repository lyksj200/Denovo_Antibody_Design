# src/predict.py

import os
import torch
import numpy as np
import argparse
import warnings
from Bio.PDB import PDBParser, DSSP, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch_geometric.data import Data

# Import from our project files
from src.config import (
    DEVICE, CHECKPOINT_DIR, GRAPH_EDGE_DISTANCE_THRESHOLD, NODE_FEATURE_DIM, TOP_K_PREDICTIONS
)
from src.models import EpitopeGNN
from src.utils import get_alpha_carbon, AMINO_ACIDS
# We need the feature calculation logic from preprocess.py
from src.preprocess import calculate_features

warnings.simplefilter('ignore', PDBConstructionWarning)


def process_antigen_pdb(pdb_path, model_for_dssp):
    """
    Processes a single antigen PDB file into a graph data object.
    This is a modified version of the preprocessing logic for prediction.

    Args:
        pdb_path (str): Path to the input PDB file.
        model_for_dssp (Bio.PDB.Model): The Bio.PDB model object for DSSP calculation.

    Returns:
        tuple: A tuple containing (graph_data, valid_residues_list) or (None, None) on failure.
    """
    try:
        # We need DSSP to calculate SASA and SS features
        dssp = DSSP(model_for_dssp, pdb_path)
    except Exception as e:
        print(f"Error running DSSP on {pdb_path}. Make sure DSSP is installed and in your PATH.")
        print(f"Details: {e}")
        return None, None
    
    node_features = []
    valid_residues = []
    
    # We assume the first chain is the antigen, or you could specify it.
    antigen_chain = next(model_for_dssp.get_chains())
    
    for residue in antigen_chain.get_residues():
        if residue.get_resname() not in AMINO_ACIDS or not get_alpha_carbon(residue):
            continue
        
        features = calculate_features(residue, dssp)
        if features is None:
            continue
            
        node_features.append(features)
        valid_residues.append(residue)
        
    if not valid_residues:
        print("No valid residues for feature extraction found in PDB file.")
        return None, None
        
    # Build graph edges
    edge_list = []
    num_nodes = len(valid_residues)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            res_i = valid_residues[i]
            res_j = valid_residues[j]
            dist = get_alpha_carbon(res_i) - get_alpha_carbon(res_j)
            if dist < GRAPH_EDGE_DISTANCE_THRESHOLD:
                edge_list.append([i, j])
                edge_list.append([j, i])
                
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    if x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError("Feature dimension mismatch during prediction.")
        
    graph_data = Data(x=x, edge_index=edge_index)
    return graph_data, valid_residues


def save_pdb_with_scores(structure, residues, scores, output_path):
    """
    Saves a new PDB file with prediction scores in the B-factor column.
    """
    # Create a mapping from residue object to its score
    res_to_score = {res.get_id(): score for res, score in zip(residues, scores)}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id() in res_to_score:
                    score = res_to_score[residue.get_id()]
                    # Set B-factor for all atoms in the residue
                    for atom in residue:
                        # Scale score to be in a typical B-factor range (e.g., 0-100)
                        atom.set_bfactor(score * 100.0)
                else:
                    # Set B-factor to 0 for non-predicted residues
                    for atom in residue:
                        atom.set_bfactor(0.0)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"\nSaved PDB with prediction scores to: {output_path}")
    print("You can now open this file in PyMOL/Chimera and color by B-factor.")


def predict(args):
    """
    Main prediction function.
    """
    # --- 1. Load Model ---
    model = EpitopeGNN().to(DEVICE)
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Load and Process Input PDB ---
    if not os.path.exists(args.pdb_path):
        print(f"Error: Input PDB file not found at {args.pdb_path}")
        return

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antigen", args.pdb_path)
    model_obj = structure[0] # The model object needed for DSSP

    print("Processing input PDB and calculating features...")
    graph_data, residue_list = process_antigen_pdb(args.pdb_path, model_obj)
    
    if graph_data is None:
        print("Failed to process PDB file.")
        return
        
    graph_data = graph_data.to(DEVICE)

    # --- 3. Run Inference ---
    with torch.no_grad():
        outputs = model(graph_data)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

    # --- 4. Display Results ---
    results = sorted(zip(residue_list, probs), key=lambda x: x[1], reverse=True)
    
    print(f"\n--- Top {TOP_K_PREDICTIONS} Predicted Epitope Residues ---")
    print("Chain | Res ID | Res Name | Score")
    print("------------------------------------")
    for residue, score in results[:TOP_K_PREDICTIONS]:
        chain_id = residue.get_parent().id
        res_id = residue.get_id()[1]
        res_name = residue.get_resname()
        print(f"  {chain_id}   |  {res_id:<5} |   {res_name}    | {score:.4f}")
        
    # --- 5. Save new PDB if output path is provided ---
    if args.output_path:
        save_pdb_with_scores(structure, [r for r, s in results], [s for r, s in results], args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict antigen epitopes from a protein structure.")
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the input PDB file for the antigen.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional. Path to save a new PDB file with scores in the B-factor column.")
    
    args = parser.parse_args()
    predict(args)