import os
import torch
from train import prepare_data, initialize_model, train_model
from prepare_data import load_graph_data

BASE_DIR = "/content/drive/MyDrive/NBA_GNN_files/"
PLOTS_DIR = os.path.join(BASE_DIR, "plot_data")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and prepare data
graphs, labels = load_graph_data()
train_loader, val_loader, test_loader, num_node_features = prepare_data(graphs, labels, batch_size=256)

model = initialize_model(
    model_type='GAT',
    num_node_features=num_node_features,
    hidden_dim=128,          # Increase from 16 to 64
    num_heads=4,            # Increase from 3 to 4
    num_layers=2,           # Increase from 1 to 2
    num_pre_layers=2,
    num_post_layers=2,
    dropout=0.3,
    train_eps=False,
    device=device
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3)

# Define class weights (penalize made shot misclassification more)
class_weights = None#torch.tensor([1.0, 1.2], device=device)  # [miss_weight, make_weight]

# Train with class weightsx
history, test_results = train_model(
    model, train_loader, val_loader, test_loader, optimizer,
    epochs=200,
    patience=25,
    device=device,
    class_weights=class_weights,  # Pass the weights,
    model_save_path='/content/drive/MyDrive/NBA_GNN_files/models/gat_2layer(128)_model.pt'
)

# Visualize
# Prepare results dictionary
results = {
    'history': history,
    **test_results  # Unpack test_results to include all keys
}

# Visualize and save to Drive
from visualize import visualize_all
visualize_all(results, output_dir=PLOTS_DIR)

print(f"\nAll plots saved to: {PLOTS_DIR}")
