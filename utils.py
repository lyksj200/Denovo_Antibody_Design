# src/utils.py

import os
import numpy as np
from Bio.PDB import PDBParser

# --- Amino Acid Constants ---

# Standard 20 amino acids
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Mapping from 3-letter code to 1-letter code
AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# Mapping from 3-letter code to an integer index for one-hot encoding
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydrophobicity scale
# A more positive value means more hydrophobic
KYTE_DOOLITTLE_HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}


# --- PDB & Geometry Functions ---

def get_structure_from_pdb(pdb_id: str, pdb_dir: str):
    """
    Parses a PDB file and returns a Bio.PDB.Structure object.

    Args:
        pdb_id (str): The PDB identifier (e.g., '1ADQ').
        pdb_dir (str): The directory containing the PDB files.

    Returns:
        Bio.PDB.Structure.Structure: The parsed structure object, or None if file not found.
    """
    parser = PDBParser(QUIET=True)
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"Warning: PDB file not found at {pdb_path}")
        return None
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        return structure
    except Exception as e:
        print(f"Error parsing PDB file {pdb_id}: {e}")
        return None


def get_alpha_carbon(residue):
    """
    Returns the alpha-carbon (C-alpha) atom of a residue.

    Args:
        residue (Bio.PDB.Residue.Residue): The residue object.

    Returns:
        Bio.PDB.Atom.Atom: The C-alpha atom object, or None if not found.
    """
    if 'CA' in residue:
        return residue['CA']
    return None


def get_min_distance(residue, other_chain):
    """
    Calculates the minimum distance between any atom of a residue and any atom in another chain.

    Args:
        residue (Bio.PDB.Residue.Residue): The residue to calculate distance from.
        other_chain (Bio.PDB.Chain.Chain): The chain to calculate distance to.

    Returns:
        float: The minimum distance. Returns infinity if calculation is not possible.
    """
    min_dist = float('inf')
    for atom1 in residue.get_atoms():
        for other_residue in other_chain.get_residues():
            for atom2 in other_residue.get_atoms():
                # The subtraction of two Bio.PDB.Atom objects returns the distance
                distance = atom1 - atom2
                if distance < min_dist:
                    min_dist = distance
    return min_dist