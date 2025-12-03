"""
Simple GAT model for NBA shot prediction (graph classification)
"""

import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from build_graph import build_graph_from_shot
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ShotGAT(torch.nn.Module):
    """
    Simple GAT model for shot prediction.
    
    Architecture:
    - 3 GATv2 layers with multi-head attention and edge features
    - Residual connections
    - Global mean pooling
    - MLP classifier
    """
    def __init__(self, node_features=15, edge_features=7, hidden_dim=128, num_heads=8, dropout=0.2):
        super(ShotGAT, self).__init__()
        
        # GATv2 layers with edge features
        self.conv1 = GATv2Conv(node_features, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_features)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_features)
        self.conv3 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_features)
        
        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        
        # Classification head
        self.fc1 = torch.nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, 2)  # Binary: made (1) or missed (0)
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # GATv2 layers with edge features and batch norm
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification with deeper MLP
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


def load_and_prepare_data(json_path='shot_data_new.json', max_samples=None):
    """
    Load shot data and convert to PyTorch Geometric graphs.
    """
    print(f"Loading shot data from {json_path}...")
    with open(json_path, 'r') as f:
        shots = json.load(f)
    
    if max_samples:
        shots = shots[:max_samples]
    
    print(f"Converting {len(shots)} shots to graphs...")
    graphs = []
    labels = []
    
    for shot in tqdm(shots, desc="Building graphs"):
        graph = build_graph_from_shot(shot)
        if graph is not None:
            # Add label (1 = made, 0 = missed)
            label = 1 if shot['made'] else 0
            graph.y = torch.tensor([label], dtype=torch.long)
            
            graphs.append(graph)
            labels.append(label)
    
    print(f"\nSuccessfully built {len(graphs)} graphs")
    print(f"  Made: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Missed: {len(labels) - sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    
    return graphs, labels


def split_data(graphs, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Inductive split: train/val/test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs, labels, train_size=train_ratio, random_state=42, stratify=labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_graphs, test_graphs, val_labels, test_labels = train_test_split(
        temp_graphs, temp_labels, train_size=val_ratio_adjusted, random_state=42, stratify=temp_labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_graphs)} graphs ({100*sum(train_labels)/len(train_labels):.1f}% made)")
    print(f"  Val:   {len(val_graphs)} graphs ({100*sum(val_labels)/len(val_labels):.1f}% made)")
    print(f"  Test:  {len(test_graphs)} graphs ({100*sum(test_labels)/len(test_labels):.1f}% made)")
    
    return train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels


def train_epoch(model, loader, optimizer, device, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        if class_weights is not None:
            loss = F.nll_loss(out, data.y, weight=class_weights)
        else:
            loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """Evaluate model with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    accuracy = correct / total
    
    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_0_mask = all_labels == 0
    class_1_mask = all_labels == 1
    
    class_0_acc = (all_preds[class_0_mask] == all_labels[class_0_mask]).mean() if class_0_mask.sum() > 0 else 0
    class_1_acc = (all_preds[class_1_mask] == all_labels[class_1_mask]).mean() if class_1_mask.sum() > 0 else 0
    
    return total_loss / total, accuracy, class_0_acc, class_1_acc


def main():
    # Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    HIDDEN_DIM = 128
    NUM_HEADS = 8
    DROPOUT = 0.3
    USE_CLASS_WEIGHTS = True  # Balance class imbalance
    MAX_SAMPLES = None  # Set to smaller number for testing, None for all data
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    graphs, labels = load_and_prepare_data(max_samples=MAX_SAMPLES)
    
    # Split data
    train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels = split_data(graphs, labels)
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate class weights for imbalanced data
    class_weights = None
    if USE_CLASS_WEIGHTS:
        train_labels_tensor = torch.tensor(train_labels)
        class_counts = torch.bincount(train_labels_tensor)
        class_weights = len(train_labels_tensor) / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(device)
        print(f"\nClass weights: {class_weights.cpu().numpy()}")
        print(f"  Class 0 (missed): {class_weights[0]:.3f}")
        print(f"  Class 1 (made):   {class_weights[1]:.3f}")
    
    # Initialize model
    model = ShotGAT(
        node_features=15,  # x, y, z, has_ball, is_offense, dist_to_rim, min_def_dist, angle, dist_3pt, nearby_def, + 5 position encoding
        edge_features=7,   # x_rel, y_rel, distance, edge_angle, + 3 edge type one-hot
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Training GAT model")
    print(f"{'='*60}")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_loss, val_acc, val_acc_0, val_acc_1 = evaluate(model, val_loader, device)
        
        # Step the scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} (Miss: {val_acc_0:.3f}, Make: {val_acc_1:.3f})")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_gat_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break
    
    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load('best_gat_model.pt'))
    test_loss, test_acc, test_acc_0, test_acc_1 = evaluate(model, test_loader, device)
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"\nTest Results:")
    print(f"  Overall accuracy: {test_acc:.4f}")
    print(f"  Missed shots (class 0) accuracy: {test_acc_0:.4f}")
    print(f"  Made shots (class 1) accuracy:   {test_acc_1:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    
    # Calculate baseline (always predict majority class)
    baseline_acc = max(sum(labels), len(labels) - sum(labels)) / len(labels)
    print(f"\nBaseline (majority class): {baseline_acc:.4f}")
    print(f"Improvement over baseline: {(test_acc - baseline_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
