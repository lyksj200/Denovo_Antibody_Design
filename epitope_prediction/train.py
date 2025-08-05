# src/train.py

import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

# Import from our project files
from src.config import (
    DEVICE, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR
)
from src.models import EpitopeGNN
from src.dataset import create_dataloaders

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_one_epoch(model, dataloader, optimizer, criterion):
    """
    Executes a single training epoch.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion: The loss function.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    
    # Use tqdm for a progress bar
    for data in tqdm(dataloader, desc="Training"):
        data = data.to(DEVICE)  # Move data to the configured device (GPU/CPU)
        
        optimizer.zero_grad()  # Clear previous gradients
        
        # Forward pass
        outputs = model(data)
        
        # The model outputs logits, labels are in data.y
        # The loss function expects logits and labels of the same shape
        loss = criterion(outputs, data.y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion):
    """
    Evaluates the model on a given dataset (validation or test).

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion: The loss function.

    Returns:
        tuple: A tuple containing (average_loss, average_auc).
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data in tqdm(dataloader, desc="Evaluating"):
            data = data.to(DEVICE)
            
            outputs = model(data)
            loss = criterion(outputs, data.y)
            
            total_loss += loss.item() * data.num_graphs
            
            # Apply sigmoid to logits to get probabilities for AUC calculation
            preds = torch.sigmoid(outputs)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    # Concatenate predictions and labels from all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc


def main():
    """
    The main function to run the training and evaluation pipeline.
    """
    print(f"Using device: {DEVICE}")

    # --- 1. Create DataLoaders ---
    try:
        train_loader, val_loader, _ = create_dataloaders()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error creating dataloaders: {e}")
        return

    # --- 2. Initialize Model, Optimizer, and Loss Function ---
    model = EpitopeGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Use BCEWithLogitsLoss, which is numerically stable and combines sigmoid + BCELoss
    # It automatically handles the logits from our model.
    criterion = nn.BCEWithLogitsLoss()

    # --- 3. Training Loop ---
    best_val_auc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = evaluate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation AUC: {val_auc:.4f}")
        
        # --- 4. Save the best model ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved with AUC: {val_auc:.4f} to {best_model_path}")

    print("\nTraining finished.")
    print(f"Best validation AUC achieved: {best_val_auc:.4f}")
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()