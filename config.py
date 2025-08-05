# src/config.py

import torch
import os

# --- Path Settings ---
# Get the project root directory. We assume 'src' is a subdirectory of the root.
# os.path.dirname(__file__) gets the directory of the current file (src)
# os.path.dirname(...) gets the parent directory of that, which is the project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for raw PDB files (antigen-antibody complexes)
PDB_DIR = os.path.join(ROOT_DIR, 'data', 'pdb')

# Directory to store preprocessed graph data
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# Directory to store trained model checkpoints
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')


# --- Data Preprocessing Parameters ---

# Distance threshold to define an epitope residue (in Angstroms, Å).
# An antigen residue is considered part of the epitope if any of its atoms
# are within this distance to any atom of an antibody.
EPITOPE_DISTANCE_THRESHOLD = 4.5

# Maximum distance between C-alpha atoms for two residues to be considered
# connected by an edge in the protein graph (in Angstroms, Å).
GRAPH_EDGE_DISTANCE_THRESHOLD = 10.0

# Identifiers for antibody chains. You may need to adjust this list based on your PDB files.
# For example, ['H', 'L'] for heavy and light chains.
ANTIBODY_CHAIN_IDS = ['H', 'L']


# --- Model Hyperparameters ---

# Dimension of the node (residue) input features.
# This value must exactly match the number of features created in `preprocess.py`.
# Example: One-hot(20) + SASA(1) + RSA(1) + SS(3) + Hydrophobicity(1) = 26
NODE_FEATURE_DIM = 26

# Dimension of the hidden layers in the Graph Neural Network (GNN)
HIDDEN_DIM = 128

# Number of GNN layers
NUM_GNN_LAYERS = 4

# Dropout rate to prevent overfitting
DROPOUT_RATE = 0.3


# --- Training Hyperparameters ---

# Learning rate for the optimizer
LEARNING_RATE = 1e-4

# Batch size for training
BATCH_SIZE = 8

# Number of training epochs
EPOCHS = 100

# Training device: auto-detect CUDA availability, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Split ratios for training, validation, and test sets
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
# TEST_SPLIT will be calculated as 1.0 - TRAIN_SPLIT - VALIDATION_SPLIT


# --- Prediction Parameters ---
# In predict.py, output the Top K residues with the highest prediction scores
TOP_K_PREDICTIONS = 20