"""
Data loading and preprocessing functions for GAT shot prediction.
Handles loading graph data from disk and splitting into train/val/test sets.
"""

import os
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Google Drive paths
BASE_DIR = "/content/drive/MyDrive/NBA_GNN_files/"
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_data")


def load_graph_data(graph_data_dir=GRAPH_DATA_DIR, max_samples=None):
    """
    Load pre-built graph data from .pt files.
    
    Args:
        graph_data_dir (str): Directory containing graph data files
        max_samples (int): Maximum number of files to load (None = all)
    
    Returns:
        graphs (list): List of PyTorch Geometric Data objects
        labels (list): List of labels (0=missed, 1=made)
    """
    print(f"Loading graph data from {graph_data_dir}...")
    
    if not os.path.exists(graph_data_dir):
        raise FileNotFoundError(f"Graph data directory not found: {graph_data_dir}")
    
    # Get all .pt items (files or directories)
    all_items = sorted([f for f in os.listdir(graph_data_dir) if f.endswith('.pt')])
    graph_dirs = [f for f in all_items if os.path.isdir(os.path.join(graph_data_dir, f))]
    graph_files = [f for f in all_items if os.path.isfile(os.path.join(graph_data_dir, f))]
    
    if len(graph_dirs) > 0:
        print(f"Found {len(graph_dirs)} graph directories (legacy format)")
        items_to_load = graph_dirs
    elif len(graph_files) > 0:
        print(f"Found {len(graph_files)} graph files")
        items_to_load = graph_files
    else:
        raise FileNotFoundError(f"No .pt files found in {graph_data_dir}")
    
    if max_samples:
        items_to_load = items_to_load[:max_samples]
        print(f"Loading first {max_samples} files")
    
    graphs = []
    labels = []
    failed_count = 0
    
    for item_name in tqdm(items_to_load, desc="Loading graphs"):
        try:
            item_path = os.path.join(graph_data_dir, item_name)
            data_dict = torch.load(item_path, weights_only=False)
            
            # Handle dictionary format
            if isinstance(data_dict, dict) and 'graphs' in data_dict and 'labels' in data_dict:
                file_graphs = data_dict['graphs']
                file_labels = data_dict['labels']
                
                for graph, label in zip(file_graphs, file_labels):
                    graph.y = label
                    graphs.append(graph)
                    labels.append(label.item() if hasattr(label, 'item') else int(label))
            
            # Handle single Data object
            elif hasattr(data_dict, 'y'):
                label = data_dict.y.item() if data_dict.y.dim() > 0 else data_dict.y
                graphs.append(data_dict)
                labels.append(label)
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print(f"\nWarning: Failed to load {item_name}: {str(e)[:200]}")
    
    if len(graphs) == 0:
        raise ValueError(f"No valid graphs were loaded from {graph_data_dir}")
    
    print(f"\nLoaded {len(graphs)} graphs ({failed_count} failed)")
    print(f"Made shots: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"Missed shots: {len(labels)-sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    
    # Print data statistics
    first_graph = graphs[0]
    print(f"\nGraph statistics:")
    print(f"  Node features: {first_graph.x.shape[1]}")
    if first_graph.edge_attr is not None:
        print(f"  Edge features: {first_graph.edge_attr.shape[1]}")
    print(f"  Avg nodes per graph: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
    
    return graphs, labels


def split_data(graphs, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    Split data into train/val/test sets with stratification.
    
    Args:
        graphs (list): List of graph objects
        labels (list): List of labels
        train_ratio (float): Proportion for training (default: 0.7)
        val_ratio (float): Proportion for validation (default: 0.2)
        test_ratio (float): Proportion for testing (default: 0.1)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs, labels, 
        train_size=train_ratio, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_graphs, test_graphs, val_labels, test_labels = train_test_split(
        temp_graphs, temp_labels, 
        train_size=val_ratio_adjusted, 
        random_state=random_state, 
        stratify=temp_labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_graphs)} graphs ({100*sum(train_labels)/len(train_labels):.1f}% made)")
    print(f"  Val:   {len(val_graphs)} graphs ({100*sum(val_labels)/len(val_labels):.1f}% made)")
    print(f"  Test:  {len(test_graphs)} graphs ({100*sum(test_labels)/len(test_labels):.1f}% made)")
    
    return train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels

def get_data_loaders(train_graphs, val_graphs, test_graphs, batch_size=32, shuffle_train=True):
    """
    Create PyTorch Geometric DataLoaders for train/val/test sets.
    
    Args:
        train_graphs (list): Training graphs
        val_graphs (list): Validation graphs
        test_graphs (list): Test graphs
        batch_size (int): Batch size (default: 32)
        shuffle_train (bool): Shuffle training data (default: True)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Data loading module for GAT shot prediction")
    print("\nExample usage:")
    print("  from data import load_graph_data, split_data, get_data_loaders")
    print()
    print("  # Load data")
    print("  graphs, labels = load_graph_data()")
    print()
    print("  # Split data")
    print("  train_g, val_g, test_g, train_l, val_l, test_l = split_data(graphs, labels)")
    print()
    print("  # Create data loaders")
    print("  train_loader, val_loader, test_loader = get_data_loaders(train_g, val_g, test_g)")
