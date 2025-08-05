# src/evaluate.py

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# Import from our project files
from src.config import DEVICE, CHECKPOINT_DIR
from src.models import EpitopeGNN
from src.dataset import create_dataloaders

def evaluate_model():
    """
    Loads the best trained model and evaluates its performance on the test set.
    """
    print("--- Starting Final Model Evaluation on Test Set ---")
    print(f"Using device: {DEVICE}")

    # --- 1. Load Test Data ---
    # We only need the test_loader for final evaluation.
    try:
        _, _, test_loader = create_dataloaders()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error creating dataloaders: {e}")
        return

    # --- 2. Load the Best Trained Model ---
    model = EpitopeGNN().to(DEVICE)
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please run train.py first to train and save a model.")
        return

    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set the model to evaluation mode (very important!)

    # --- 3. Perform Inference on the Test Set ---
    all_labels = []
    all_probs = []

    with torch.no_grad():  # Disable gradient calculations
        for data in test_loader:
            data = data.to(DEVICE)
            outputs = model(data)
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities

            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    # Concatenate results from all batches
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()

    # --- 4. Calculate and Print Metrics ---
    print("\n--- Performance Metrics on Test Set ---")

    # AUC (Area Under the ROC Curve)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Area Under ROC Curve (AUC): {auc:.4f}")

    # AUPRC (Area Under the Precision-Recall Curve)
    auprc = average_precision_score(all_labels, all_probs)
    print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")

    # Threshold-based metrics
    # We use a standard 0.5 threshold to convert probabilities to binary predictions
    print("\nMetrics based on a 0.5 probability threshold:")
    binary_preds = (all_probs >= 0.5).astype(int)
    
    # Generate a detailed classification report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, binary_preds, target_names=['Non-Epitope (0)', 'Epitope (1)'])
    print(report)
    
    print("-----------------------------------------")


if __name__ == "__main__":
    evaluate_model()