#
# -----------------
#
# ## 1. Setup and Imports
#
#
#
import os, math, gc
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv, LayerNorm

# Simplified threading settings for better stability
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

#
# -----------------
#
# ## 2. Reproducibility
#
#
#
def set_seed(seed: int = 42):
    """Sets a random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

#
# -----------------
#
# ## 3. Load and Prepare Data
#
#
#
# Define data paths
TRAIN_DATA_PATH = "data/train_set.parquet"
TEST_DATA_PATH = "data/test_set.parquet"

# Load the datasets from Parquet files
if not (os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH)):
    raise FileNotFoundError(
        f"Could not find data splits. Please ensure '{TRAIN_DATA_PATH}' and '{TEST_DATA_PATH}' exist."
    )

train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)

print("Loaded train data:", TRAIN_DATA_PATH, "shape=", train_df.shape)
print("Loaded test data:", TEST_DATA_PATH, "shape=", test_df.shape)

# Define column names
SRC_COL = 'Src IP'
DST_COL = 'Dst IP'
LABEL_COL = 'target'
EDGE_COLS = [
    'Bwd Packet Length Min', 'Protocol_6', 'Bwd Packets/s', 'FWD Init Win Bytes',
    'Packet Length Std', 'FIN Flag Count', 'SrcPortRange_registered',
    'Packet Length Min', 'Fwd Seg Size Min', 'DstPortRange_well_known',
    'Bwd IAT Total', 'SYN Flag Count', 'Bwd Packet Length Std'
]

print(f"Using columns -> src: {SRC_COL}, dst: {DST_COL}, label: {LABEL_COL}")
print(f"Using {len(EDGE_COLS)} edge feature columns.")

#
# -----------------
#
# ### Data Splitting (80-20 Train-Test Split)
#
#
#
# Data is pre-split, so we just report the shapes and class ratios.
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Train set positive class ratio: {train_df[LABEL_COL].mean():.4f}")
print(f"Test set positive class ratio: {test_df[LABEL_COL].mean():.4f}")

#
# -----------------
#
# ## 4. Static Graph Construction
#
#
#
def pick_edge_feature_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Selects numeric columns to be used as edge features."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

def aggregate_edges(df: pd.DataFrame, src_col: str, dst_col: str, label_col: str, edge_cols: List[str]) -> pd.DataFrame:
    """Aggregates multiple interactions into unique edges."""
    g = df[[src_col, dst_col] + edge_cols + [label_col]]
    
    # Aggregate features by taking multiple statistics to capture distribution
    if edge_cols:
        agg_funcs = ['mean', 'std', 'min', 'max']
        feat_df = g.groupby([src_col, dst_col], as_index=False)[edge_cols].agg(agg_funcs)
        # Fill NaNs that result from std on single-item groups
        feat_df = feat_df.fillna(0)
        # Flatten the multi-level column names (e.g., ('Bwd Packets/s', 'mean') -> 'Bwd Packets/s_mean')
        feat_df.columns = ['_'.join(col).strip('_') for col in feat_df.columns.values]
    else:
        feat_df = g[[src_col, dst_col]].drop_duplicates()
    
    # Aggregate labels: if any interaction is malicious, the edge is labeled as malicious
    lbl_df = g.groupby([src_col, dst_col], as_index=False)[label_col].max()
    
    # The new column names for merging are the original src/dst columns
    merge_on = [src_col, dst_col]
    return pd.merge(feat_df, lbl_df, on=merge_on)

def build_static_graph(
    df_part: pd.DataFrame,
    src_col: str,
    dst_col: str,
    label_col: str,
    edge_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[Data, dict, StandardScaler]:
    """Builds a PyG Data object from a dataframe."""
    agg_df = aggregate_edges(df_part, src_col, dst_col, label_col, edge_cols)

    # After aggregation, get the new list of feature columns for scaling
    agg_feature_cols = [c for c in agg_df.columns if c not in [src_col, dst_col, label_col]]

    # Create a unified mapping from IP addresses to integer indices
    ips = pd.Index(pd.unique(pd.concat([agg_df[src_col], agg_df[dst_col]])))
    ip2idx = {ip: i for i, ip in enumerate(ips)}
    src = agg_df[src_col].map(ip2idx).values
    dst = agg_df[dst_col].map(ip2idx).values
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # Standardize edge features using the new aggregated column names
    if fit_scaler:
        scaler = StandardScaler()
        edge_features = scaler.fit_transform(agg_df[agg_feature_cols])
    else:
        edge_features = scaler.transform(agg_df[agg_feature_cols])
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    # Edge labels
    y = torch.tensor(agg_df[label_col].astype(int).values, dtype=torch.long)
    
    # Node features: log-transformed in-degree, out-degree, and total degree
    num_nodes = len(ip2idx)
    in_deg = np.bincount(dst, minlength=num_nodes)
    out_deg = np.bincount(src, minlength=num_nodes)
    node_features = np.log1p(np.stack([in_deg, out_deg, in_deg + out_deg], axis=1))
    x = torch.tensor(node_features, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data, ip2idx, scaler

# Define which columns to use as edge features
print(f"Using {len(EDGE_COLS)} edge feature columns.")

# Build graphs
train_g, train_ip2idx, edge_scaler = build_static_graph(train_df, SRC_COL, DST_COL, LABEL_COL, EDGE_COLS, fit_scaler=True)
test_g, _, _ = build_static_graph(test_df, SRC_COL, DST_COL, LABEL_COL, EDGE_COLS, scaler=edge_scaler)

print(f"Train graph: {train_g}")
print(f"Test graph:  {test_g}")

#
# -----------------
#
# ## 5. GNN Model Architecture
#
#
#
class NodeEncoder(nn.Module):
    """GNN encoder to produce node embeddings."""
    def __init__(self, in_node: int, hidden: int, num_layers: int, gnn_type: str = "gcn", heads: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_channels = in_node
        for _ in range(num_layers):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(in_channels, hidden))
            elif gnn_type == "gat":
                self.convs.append(GATv2Conv(in_channels, hidden // heads, heads=heads))
            self.norms.append(LayerNorm(hidden))
            in_channels = hidden

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(norm(x))
            x = self.dropout(x)
        return x

class StaticEdgeClassifier(nn.Module):
    """Predicts edge labels based on node embeddings and edge features."""
    def __init__(self, in_node: int, in_edge: int, hidden: int = 64, num_layers: int = 2, gnn_type: str = "gcn"):
        super().__init__()
        self.encoder = NodeEncoder(in_node, hidden, num_layers, gnn_type=gnn_type)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + in_edge, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )

    def forward(self, data: Data):
        h = self.encoder(data.x, data.edge_index)
        src, dst = data.edge_index
        edge_input = torch.cat([h[src], h[dst], data.edge_attr], dim=1)
        return self.edge_mlp(edge_input)

#
# -----------------
#
# ## 6. Training and Evaluation
#
#
#
def train_and_evaluate(model: nn.Module, train_data: Data, test_data: Data, epochs: int = 10):
    """Handles the training and evaluation loop."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Calculate class weights to handle imbalance
    pos_weight = (train_data.y == 0).sum() / max((train_data.y == 1).sum(), 1)
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    print(f"\n--- Training {model.encoder.convs[0].__class__.__name__} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = loss_fn(logits, train_data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        y_true = test_data.y.cpu().numpy()
        
        pr_auc = average_precision_score(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        
        print(f"\nTest PR-AUC: {pr_auc:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        
    # Clean up memory
    del train_data, test_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {'pr_auc': pr_auc, 'roc_auc': roc_auc}

summary = {}


in_node_dim = train_g.x.size(1)
in_edge_dim = train_g.edge_attr.size(1)

# Train and evaluate GCN with increased capacity
gcn_model = StaticEdgeClassifier(in_node_dim, in_edge_dim, hidden=128, num_layers=3, gnn_type='gcn')
gcn_results = train_and_evaluate(gcn_model, train_g, test_g, epochs=10)
summary['gcn'] = {
        'test_pr_auc': gcn_results['pr_auc'],
        'test_roc_auc': gcn_results['roc_auc'],
        'params': sum(p.numel() for p in gcn_model.parameters())
    }
# Explicitly clear memory before training the next model
del gcn_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Train and evaluate GAT. Reduce hidden size to fit in memory.
gat_model = StaticEdgeClassifier(in_node_dim, in_edge_dim, hidden=64, num_layers=2, gnn_type='gat')
gat_results = train_and_evaluate(gat_model, train_g, test_g, epochs=10)

#
# -----------------
#
# ## 7. Summary of Results
#
#
#
summary['gat'] = {
        'test_pr_auc': gat_results['pr_auc'],
        'test_roc_auc': gat_results['roc_auc'],
        'params': sum(p.numel() for p in gat_model.parameters())
    }

print("\n--- Final Summary ---")
import json
print(json.dumps(summary, indent=2))

# To save the results to a file:
# with open('static_baseline_results.json', 'w') as f:
#     json.dump(summary, f, indent=2)