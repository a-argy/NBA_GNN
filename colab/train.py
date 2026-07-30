"""
Training and evaluation functions for GNN shot prediction models (GAT and GINE).
Handles training loop, validation, and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from gat import GATShotPredictor
from gine import GINEShotPredictor
from prepare_data import load_graph_data, split_data, get_data_loaders


# Google Drive paths
BASE_DIR = "/content/drive/MyDrive/NBA_GNN_files/"
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_epoch(model, loader, optimizer, device, class_weights=None):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        avg_loss (float): Average loss for the epoch
        accuracy (float): Training accuracy
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        # Apply class weights if provided
        if class_weights is not None:
            loss = F.nll_loss(out, data.y, weight=class_weights)
        else:
            loss = F.nll_loss(out, data.y)
        #loss = F.nll_loss(out, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        device: Device to evaluate on
    
    Returns:
        avg_loss (float): Average loss
        accuracy (float): Overall accuracy
        class_0_acc (float): Accuracy on missed shots
        class_1_acc (float): Accuracy on made shots
        all_preds (np.array): All predictions
        all_labels (np.array): All true labels
        all_probs (np.array): All prediction probabilities for positive class
    """
    model.eval()
    
    all_predictions = []
    all_labels_list = []
    
    with torch.no_grad():
        # Run model on each graph
        for data in loader:
            data = data.to(device)
            predictions = model(data)
            all_predictions.append(predictions)
            all_labels_list.append(data.y)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels_list, dim=0)
    
    # Compute loss
    loss = F.nll_loss(all_predictions, all_labels_tensor)
    
    # Compute accuracy
    pred = all_predictions.argmax(dim=1)
    correct = (pred == all_labels_tensor).sum().item()
    accuracy = correct / len(all_labels_tensor)
    
    # Convert to numpy
    all_preds = pred.cpu().numpy()
    all_labels = all_labels_tensor.cpu().numpy()
    all_probs = torch.exp(all_predictions[:, 1]).cpu().numpy()
    
    # Calculate per-class accuracy
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1
    
    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).mean() if class_0_mask.sum() > 0 else 0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).mean() if class_1_mask.sum() > 0 else 0
    
    return loss.item(), accuracy, class_0_acc, class_1_acc, all_preds, all_labels, all_probs


def prepare_data(graphs, labels, batch_size=32):
    """
    Prepare train/val/test dataloaders from graphs and labels.
    
    Args:
        graphs: List of graph objects
        labels: List of labels
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, num_node_features
    """
    train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels = split_data(graphs, labels)
    train_loader, val_loader, test_loader = get_data_loaders(
        train_graphs, val_graphs, test_graphs, batch_size, shuffle_train=False
    )
    num_node_features = train_graphs[0].x.shape[1]
    
    print(f"\nData prepared:")
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val:   {len(val_graphs)} graphs")
    print(f"  Test:  {len(test_graphs)} graphs")
    print(f"  Node features: {num_node_features}")
    
    return train_loader, val_loader, test_loader, num_node_features


def initialize_model(model_type, num_node_features, hidden_dim, num_heads, num_layers, 
                     num_pre_layers, num_post_layers, dropout, train_eps, device):
    """Initialize model based on type."""
    if model_type.upper() == 'GAT':
        model = GATShotPredictor(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            num_post_layers=num_post_layers,
            dropout=dropout
        ).to(device)
        model_desc = f"GAT with {num_pre_layers} pre + {num_layers} conv + {num_post_layers} post layers, {hidden_dim}-dim, {num_heads} heads"
    elif model_type.upper() == 'GINE':
        model = GINEShotPredictor(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            num_post_layers=num_post_layers,
            dropout=dropout,
            train_eps=train_eps
        ).to(device)
        model_desc = f"GINE with {num_pre_layers} pre + {num_layers} conv + {num_post_layers} post layers, {hidden_dim}-dim, train_eps={train_eps}"
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'GAT' or 'GINE'")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_desc}")
    print(f"Total parameters: {total_params:,}")
    
    return model


def compute_test_metrics(test_preds, test_labels, test_probs):
    """Compute test set metrics."""
    return {
        'roc_auc': roc_auc_score(test_labels, test_probs),
        'f1': f1_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds),
        'recall': recall_score(test_labels, test_preds)
    }


def train_model(model, train_loader, val_loader, test_loader, optimizer,
                epochs=100,
                patience=25,
                device=None, class_weights=None,
                model_save_path=None):
    """
    Train a GNN model with early stopping.
    
    Args:
        model: Initialized PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        optimizer: Optimizer
        epochs (int): Maximum training epochs
        patience (int): Early stopping patience
        device: Device to train on
        model_save_path (str): Path to save best model
    
    Returns:
        history (dict): Training history
        test_results (dict): Test set results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_save_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_save_path = os.path.join(MODELS_DIR, 'best_model.pt')
    
    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"\nUsing class weights: {class_weights.cpu().numpy()}")
    
    print(f"\nModel will be saved to: {model_save_path}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training Shot Predictor")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_acc_0': [], 'val_acc_1': []}
    
    for epoch in range(1, epochs + 1):
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_loss, val_acc, val_acc_0, val_acc_1, _, _, _ = evaluate(model, val_loader, device)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_acc_0'].append(val_acc_0)
        history['val_acc_1'].append(val_acc_1)
        
        print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} (Miss={val_acc_0:.3f}, Make={val_acc_1:.3f})")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best: epoch {best_epoch})")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc, test_acc_0, test_acc_1, test_preds, test_labels, test_probs = evaluate(model, test_loader, device)
    test_metrics = compute_test_metrics(test_preds, test_labels, test_probs)
    
    # Calculate baseline from test labels
    baseline_acc = max(sum(test_labels), len(test_labels) - sum(test_labels)) / len(test_labels)
    
    print(f"\nBest validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"\nTest Results:")
    print(f"  Overall accuracy: {test_acc:.4f}")
    print(f"  Missed shots (class 0): {test_acc_0:.4f}")
    print(f"  Made shots (class 1): {test_acc_1:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"\nClassification Metrics:")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"\nBaseline (majority class): {baseline_acc:.4f}")
    
    test_results = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_acc_0': test_acc_0,
        'test_acc_1': test_acc_1,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'test_probs': test_probs,
        'test_metrics': test_metrics,
        'baseline_acc': baseline_acc,
        'model_save_path': model_save_path
    }
    
    return history, test_results


def main():
    """Example of clean training workflow."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    MODEL_TYPE = 'GAT'  # 'GAT' or 'GINE'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    HIDDEN_DIM = 16
    NUM_HEADS = 4  # For GAT
    NUM_LAYERS = 2  # Number of conv layers
    DROPOUT = 0.3
    TRAIN_EPS = True  # For GINE
    MAX_SAMPLES = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load and prepare data
    graphs, labels = load_graph_data(max_samples=MAX_SAMPLES)
    train_loader, val_loader, test_loader, num_node_features = prepare_data(graphs, labels, BATCH_SIZE)
    
    # 2. Initialize model
    model = initialize_model(MODEL_TYPE, num_node_features, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, TRAIN_EPS, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 3. Train
    model_save_path = os.path.join(MODELS_DIR, f'best_{MODEL_TYPE.lower()}_model.pt')
    history, test_results = train_model(
        model, train_loader, val_loader, test_loader, optimizer,
        epochs=EPOCHS,
        patience=25,
        device=device,
        model_save_path=model_save_path
    )
    
    print("\nTraining complete! Use visualize.py to plot results.")
    
    return model, history, test_results


if __name__ == "__main__":
    main()
