import os
import torch
from train import prepare_data, initialize_model, train_model
from prepare_data import load_graph_data

BASE_DIR = "/content/drive/MyDrive/NBA_GNN_files/"
PLOTS_DIR = os.path.join(BASE_DIR, "plot_data")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load and prepare data
graphs, labels = load_graph_data()
train_loader, val_loader, test_loader, num_node_features = prepare_data(
    graphs, labels,
    batch_size=512
)

# 2. Initialize GINE model
model = initialize_model(
    model_type='GINE',           # Specify GINE
    num_node_features=num_node_features,
    hidden_dim=32,               # Hidden layer size
    num_heads=None,              # Not used for GINE (only GAT uses this)
    num_layers=2,
    num_pre_layers=1,
    num_post_layers=2,# Number of GINE conv layers
    dropout=0.3,                 # Dropout rate
    train_eps=True,              # GINE-specific: learn epsilon parameter
    device=device
)

# 3. Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# 4. Train
history, test_results = train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    epochs=200,
    patience=100,
    device=device,
    model_save_path='/content/drive/MyDrive/NBA_GNN_files/models/gine_1layer_model.pt'
)

# Visualize
# Prepare results dictionary
"""results = {
    'history': history,
    **test_results  # Unpack test_results to include all keys
}

# Visualize and save to Drive
from visualize import visualize_all
visualize_all(results, output_dir=PLOTS_DIR)

print(f"\nAll plots saved to: {PLOTS_DIR}")"""