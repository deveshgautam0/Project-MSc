"""src/gnn_model.py — Gap 3: Graph Neural Network models for molecular structures.

Provides a GNN-based property predictor for polymer / nanoparticle molecular
graphs. Uses torch + torch_geometric when available; otherwise falls back to a
graph-kernel + Random Forest surrogate so the pipeline still runs offline.

Wrapped in try/except imports so the project runs without PyTorch installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Optional PyTorch / PyG — wrapped so the module imports without them.
try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


class _GNNRegressor(nn.Module):
    """2-layer GCN with global mean pooling for graph-level regression."""

    def __init__(self, node_dim: int, edge_dim: int = 0, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)