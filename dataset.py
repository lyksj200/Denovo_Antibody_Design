# src/dataset.py

import os
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from src.config import PROCESSED_DIR, BATCH_SIZE, TRAIN_SPLIT, VALIDATION_SPLIT


class ProteinGraphDataset(Dataset):
    """
    A PyTorch Dataset class to load preprocessed protein graph data.
    It uses lazy loading, meaning it only loads data from disk when requested.
    """
    def __init__(self, data_file_list):
        """
        Args:
            data_file_list (list): A list of full paths to the .pt files.
        """
        self.data_files = data_file_list

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Loads and returns a single graph data object from the disk.

        Args:
            idx (int): The index of the data sample.

        Returns:
            torch_geometric.data.Data: The loaded graph data object.
        """
        data_path = self.data_files[idx]
        # torch.load reads a file saved with torch.save
        graph_data = torch.load(data_path)
        return graph_data


def create_dataloaders(batch_size=BATCH_SIZE, seed=42):
    """
    Scans the processed data directory, splits the data into train, validation,
    and test sets, and creates a DataLoader for each.

    Args:
        batch_size (int): The number of graphs in each batch.
        seed (int): Random seed for shuffling to ensure reproducibility.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed data directory is empty or does not exist: {PROCESSED_DIR}\n"
            "Please run preprocess.py first."
        )

    all_files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith('.pt')]
    
    # Shuffle for random splitting
    random.seed(seed)
    random.shuffle(all_files)

    # Calculate split sizes
    num_files = len(all_files)
    train_end = int(TRAIN_SPLIT * num_files)
    val_end = train_end + int(VALIDATION_SPLIT * num_files)

    # Split the file list
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    if not train_files:
        raise ValueError("Training set is empty. Check your data or split ratios.")
    if not val_files:
        raise ValueError("Validation set is empty. Check your data or split ratios.")
    if not test_files:
        raise ValueError("Test set is empty. Check your data or split ratios.")


    # Create Dataset objects
    train_dataset = ProteinGraphDataset(train_files)
    val_dataset = ProteinGraphDataset(val_files)
    test_dataset = ProteinGraphDataset(test_files)

    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoader objects using torch_geometric's DataLoader
    # This loader correctly handles batching of graph data.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # This is for testing purposes to see if the dataloaders are created correctly.
    print("Attempting to create dataloaders...")
    try:
        train_dl, val_dl, test_dl = create_dataloaders(batch_size=4)
        
        print(f"\nSuccessfully created dataloaders.")
        print(f"Number of batches in train_loader: {len(train_dl)}")
        
        # Inspect a single batch
        first_batch = next(iter(train_dl))
        print("\n--- First Batch Info ---")
        print(f"Type of batch: {type(first_batch)}")
        print(f"Batch keys: {first_batch.keys}")
        print(f"Number of graphs in batch: {first_batch.num_graphs}")
        print(f"Node features shape: {first_batch.x.shape}")
        print(f"Labels shape: {first_batch.y.shape}")
        print("------------------------")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")