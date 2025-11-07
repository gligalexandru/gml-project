import os, math, numpy as np, pandas as pd
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import read_csv
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import networkx as nx
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

EDGE_COLS = [
    'Bwd Packet Length Min', 'Protocol_6', 'Bwd Packets/s', 'FWD Init Win Bytes',
    'Packet Length Std', 'FIN Flag Count', 'SrcPortRange_registered',
    'Packet Length Min', 'Fwd Seg Size Min', 'DstPortRange_well_known',
    'Bwd IAT Total', 'SYN Flag Count', 'Bwd Packet Length Std'
]
ID_COLS = ['Src IP','Dst IP','Timestamp']
LABEL_COL = 'target'

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# train_df = pd.read_csv('data/data_gml/train.csv')
# test_df = pd.read_csv('data/data_gml/test.csv')

with open("data/snapshots_data.pkl", "rb") as f:
    saved = pickle.load(f)

train_snaps = saved["train_snaps"]
test_snaps = saved["test_snaps"]
scaler_edge = saved["scaler_edge"]
train_ip2idx = saved["train_ip2idx"]
test_ip2idx = saved["test_ip2idx"]
edge_cols_used = saved["edge_cols_used"]

class EdgeGraphSAGEConv(MessagePassing):
    """
    Edge-aware GraphSAGE (E-GraphSAGE-like):
      m_ij = gate([x_i, x_j, e_ij]) * φ( Wj x_j + Wi x_i + We e_ij )
      h_i' = Norm( mean_j m_ij + Wself x_i )     (residual)
    """
    def __init__(self, in_channels: int, edge_in: int, out_channels: int, aggr: str = "mean", dropout: float = 0.0):
        super().__init__(aggr=aggr, node_dim=0)
        self.lin_src  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_in,    out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

        # Small gate that decides how much of each message passes through
        self.gate = nn.Sequential(
            nn.Linear(in_channels + in_channels + edge_in, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1)
        )

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # -> [N, Fout]
        out = out + self.lin_self(x)  # residual
        out = self.norm(out)
        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        msg_raw = self.lin_src(x_j) + self.lin_dst(x_i) + self.lin_edge(edge_attr)
        g = torch.sigmoid(self.gate(torch.cat([x_i, x_j, edge_attr], dim=-1)))
        msg = F.relu(msg_raw) * g
        return self.dropout(msg)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out

class TemporalEdgeSAGEClassifier(nn.Module):
    """
    GraphSAGE over each snapshot + GRUCell over node states across time.
    Compatible with your EdgeGraphSAGEConv and edge-level head.
    """
    def __init__(self, in_node: int, in_edge: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden = hidden                    # <-- expose hidden
        self.dropout = nn.Dropout(dropout)

        # --- spatial backbone (edge-aware GraphSAGE) ---
        from torch_geometric.nn import MessagePassing

        class EdgeGraphSAGEConv(MessagePassing):
            def __init__(self, in_channels, edge_in, out_channels, aggr="mean", dropout=0.0):
                super().__init__(aggr=aggr, node_dim=0)
                self.lin_src  = nn.Linear(in_channels, out_channels, bias=False)
                self.lin_dst  = nn.Linear(in_channels, out_channels, bias=False)
                self.lin_edge = nn.Linear(edge_in,    out_channels, bias=False)
                self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
                self.gate = nn.Sequential(
                    nn.Linear(in_channels + in_channels + edge_in, out_channels // 2),
                    nn.ReLU(),
                    nn.Linear(out_channels // 2, 1)
                )
                self.norm = nn.LayerNorm(out_channels)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, edge_index, edge_attr):
                out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
                out = out + self.lin_self(x)
                out = self.norm(out)
                return out

            def message(self, x_i, x_j, edge_attr):
                msg_raw = self.lin_src(x_j) + self.lin_dst(x_i) + self.lin_edge(edge_attr)
                g = torch.sigmoid(self.gate(torch.cat([x_i, x_j, edge_attr], dim=-1)))
                msg = F.relu(msg_raw) * g
                return self.dropout(msg)

            def update(self, aggr_out):
                return aggr_out

        layers = []
        dims = [in_node] + [hidden] * num_layers
        for i in range(num_layers):
            layers.append(EdgeGraphSAGEConv(dims[i], in_edge, dims[i+1], aggr="mean", dropout=dropout))
        self.convs = nn.ModuleList(layers)

        # --- temporal cell over nodes ---
        self.gru = nn.GRUCell(hidden, hidden)

        # --- edge head (uses recurrent node states) ---
        edge_head_in = (2 * hidden) + in_edge + hidden + hidden  # [h_s, h_d, e, |h_s-h_d|, h_s*h_d]
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

    @torch.no_grad()
    def init_state(self, num_nodes: int, device=None):
        """Create zero initial node states of shape [num_nodes, hidden]."""
        return torch.zeros(num_nodes, self.hidden, device=device)

    def spatial_encode(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
        return x  # [N, hidden]

    def forward(self, data: Data, h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (edge_logits, h_t)
        """
        x_star = self.spatial_encode(data)                     # [N, H]
        if h_prev is None:
            h_prev = x_star.new_zeros(x_star.size(0), x_star.size(1))
        h_t = self.gru(x_star, h_prev)                         # [N, H]

        src, dst = data.edge_index
        h_s, h_d = h_t[src], h_t[dst]
        h_abs = torch.abs(h_s - h_d)
        h_mul = h_s * h_d
        z = torch.cat([h_s, h_d, data.edge_attr, h_abs, h_mul], dim=-1)
        logits = self.edge_mlp(z)
        return logits, h_t

def run_epoch_fullgraph(model, snapshots, optimizer=None, device='cpu'):
    is_train = optimizer is not None
    total_loss, total_correct, total_edges = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()
    all_preds, all_trues = [], []

    for data in snapshots:
        data = data.to(device)
        logits = model(data)
        if is_train:
            loss = ce(logits, data.y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * data.y.numel()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(data.y.cpu().numpy())
            total_correct += int((pred == data.y).sum())
            total_edges += int(data.y.numel())

    # Metrics
    if all_trues:
        y_true = np.concatenate(all_trues)
        y_pred = np.concatenate(all_preds)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        # FPR: FP / N_negatives
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        fpr = fp / max(1, (fp + tn))
    else:
        weighted_f1, fpr = float('nan'), float('nan')

    avg_loss = (total_loss / max(1, total_edges)) if is_train else None
    acc = total_correct / max(1, total_edges)
    return avg_loss, acc, weighted_f1, fpr

class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.eps = gamma_pos, gamma_neg, eps
    def forward(self, logits, y):
        p = torch.softmax(logits, dim=1)[:,1]
        yf = y.float()
        pt = p*yf + (1-p)*(1-yf)
        gamma = self.gp*yf + self.gn*(1-yf)
        loss = - (yf*torch.log(p+self.eps) + (1-yf)*torch.log(1-p+self.eps)) * ((1-pt)**gamma)
        return loss.mean()

def run_epoch_neighbor_temporal(
    model,
    snapshots,
    optimizer=None,
    device='cuda',
    num_neighbors=[25, 10],
    batch_size=4096,
    shuffle=True,
    clip=2.0,
    tbptt=5,                 # detach global state every N mini-batches to cap memory
):
    """
    If optimizer is None -> eval mode (no loss/updates, returns avg_loss=None).
    Maintains a global H over nodes; per mini-batch we read/write H[batch.n_id].
    """
    is_train = optimizer is not None
    ce = nn.CrossEntropyLoss()
    model.train() if is_train else model.eval()

    # All snapshots in a split share the same node index space
    N = snapshots[0].x.size(0)
    H_global = torch.zeros(N, model.hidden, device=device)

    total_loss, total_correct, total_edges = 0.0, 0, 0
    all_preds, all_trues = [], []
    steps_since_detach = 0

    # Process snapshots in chronological order
    for snap in sorted(snapshots, key=lambda d: getattr(d, "_bin", 0)):
        loader = NeighborLoader(
            snap, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle
        )
        for batch in loader:
            batch = batch.to(device)

            # Map global node states -> batch node states
            h_prev_batch = H_global[batch.n_id]                         # [N_batch_nodes, H]
            logits, h_t_batch = model(batch, h_prev=h_prev_batch)       # (edge_logits, new_node_states)

            if is_train:
                loss = ce(logits, batch.y)
                if torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    total_loss += float(loss.item()) * batch.y.numel()

            with torch.no_grad():
                pred = logits.argmax(1)
                all_preds.append(pred.detach().cpu().numpy())
                all_trues.append(batch.y.detach().cpu().numpy())
                total_correct += int((pred == batch.y).sum())
                total_edges   += int(batch.y.numel())

                # Write back updated node states for nodes seen in this subgraph
                H_global[batch.n_id] = h_t_batch.detach()

            # Truncated BPTT: periodically detach the whole global state
            steps_since_detach += 1
            if is_train and steps_since_detach >= tbptt:
                H_global = H_global.detach()
                steps_since_detach = 0

    # Metrics
    if all_trues:
        y_true = np.concatenate(all_trues)
        y_pred = np.concatenate(all_preds)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / max(1, (fp + tn))
        acc = (y_pred == y_true).mean()
    else:
        weighted_f1, fpr, acc = float('nan'), float('nan'), float('nan')

    avg_loss = (total_loss / max(1, total_edges)) if is_train else None
    return avg_loss, acc, weighted_f1, fpr


def optimize_numeric_dtypes(df: pd.DataFrame, try_float16: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest possible dtype without changing values.
    - Integers: downcast to smallest signed/unsigned integer.
    - Floats: downcast to float32 (and optionally float16 if lossless within tolerance).
    Returns a new DataFrame (original unchanged).
    """
    result = df.copy()
    start_mem = result.memory_usage(deep=True).sum() / 1024**2

    num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    for c in num_cols:
        col = result[c]

        # Skip all-NaN
        if col.notnull().sum() == 0:
            continue

        if pd.api.types.is_integer_dtype(col):
            # Integer (no NaNs)
            if col.min() >= 0:
                result[c] = pd.to_numeric(col, downcast="unsigned")
            else:
                result[c] = pd.to_numeric(col, downcast="integer")

        elif pd.api.types.is_float_dtype(col):
            # First, try float32
            col32 = col.astype(np.float32)
            if np.allclose(col.values, col32.values, equal_nan=True):
                result[c] = col32
                # Optionally try float16 (more aggressive)
                if try_float16:
                    col16 = col.astype(np.float16)
                    if np.allclose(col.values, col16.astype(np.float32).values, rtol=1e-03, atol=1e-06, equal_nan=True):
                        result[c] = col16
            # else keep original float64

        # If it's a nullable integer (Int64/Int32), try to preserve nulls with the smallest nullable int
        elif pd.api.types.is_dtype_equal(col.dtype, "Int64") or str(col.dtype).startswith("Int"):
            if col.min() >= 0:
                tmp = pd.to_numeric(col.astype("float64"), downcast="unsigned")
            else:
                tmp = pd.to_numeric(col.astype("float64"), downcast="integer")
            # Cast back to nullable integer if still integer-like
            if pd.api.types.is_integer_dtype(tmp):
                result[c] = pd.Series(tmp, index=col.index).astype(pd.ArrowDtype(tmp.dtype.name) if hasattr(pd, "ArrowDtype") else tmp.dtype)

    end_mem = result.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start_mem:.2f} MB → {end_mem:.2f} MB ({(start_mem-end_mem):.2f} MB saved, {(1 - end_mem/max(start_mem,1e-9))*100:.1f}% reduction)")

    return result


@torch.no_grad()
def confusion_and_plot_temporal(model, snapshots, device='cuda', labels=(0, 1), title='Confusion Matrix (Test)'):
    """
    Evaluate a temporal GNN across snapshots in chronological order,
    carrying node state across time. Plots and returns the confusion matrix.
    """
    model.eval()

    # Sort by time and init recurrent state
    snaps = sorted(snapshots, key=lambda d: getattr(d, "_bin", 0))
    N0 = snaps[0].x.size(0)
    hidden_dim = getattr(model, "hidden", None) or model.gru.hidden_size
    H = torch.zeros(N0, hidden_dim, device=device)

    all_preds, all_labels = [], []

    for snap in snaps:
        snap = snap.to(device)

        # If node count differs, re-init state (e.g., different split)
        if H.size(0) != snap.x.size(0):
            H = torch.zeros(snap.x.size(0), hidden_dim, device=device)

        logits, H = model(snap, H)  # temporal forward
        preds  = logits.argmax(dim=1).cpu().numpy()
        labels_np = snap.y.cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels_np)

        H = H.detach()  # truncate (safety)

    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    report = classification_report(y_true, y_pred, labels=list(labels), digits=4, zero_division=0)

    # --- Plot above the text output ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f'Pred {l}' for l in labels],
        yticklabels=[f'True {l}' for l in labels]
    )
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout()
    plt.show()

    print("📊 Classification Report:")
    print(report)
    return cm, report


if __name__ == "__main__":
    in_node = train_snaps[0].x.size(1)         # now includes centralities (+ optional per-bin feats)
    in_edge = train_snaps[0].edge_attr.size(1) # edge features + time enc
    model = TemporalEdgeSAGEClassifier(in_node=in_node, in_edge=in_edge,
                                    hidden=32, num_layers=2, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


    EPOCHS = 1
    for epoch in range(1, EPOCHS+1):
        # Train across ALL train snapshots (temporal state is carried within the function)
        tr_loss, tr_acc, tr_f1, tr_fpr = run_epoch_neighbor_temporal(
            model, train_snaps, optimizer=opt, device=device,
            num_neighbors=[25,10], batch_size=4096, shuffle=True, tbptt=5
        )

        # Evaluate across ALL test snapshots (no optimizer = eval)
        _, te_acc, te_f1, te_fpr = run_epoch_neighbor_temporal(
            model, test_snaps, optimizer=None, device=device,
            num_neighbors=[25,10], batch_size=4096, shuffle=False, tbptt=5
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} | train acc {tr_acc:.4f} | "
            f"train F1 {tr_f1:.4f} | train FPR {tr_fpr:.4f} | "
            f"test acc {te_acc:.4f} | test F1 {te_f1:.4f} | test FPR {te_fpr:.4f}"
        )

    cm, report = confusion_and_plot_temporal(model, test_snaps, device=device)

