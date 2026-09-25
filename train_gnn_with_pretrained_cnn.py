"""
Train a small GNN on top of a (frozen) pretrained per-electrode CNN.

Pipeline per graph:
  - Read electrode signals from x[: num_elec] (already per-file z-scored by
    preprocess_graphs.py with --scaling std)
  - Encode with the pretrained ElectrodeCNN -> (num_elec, embed_dim)
  - Initialize TM node features as one of:
      * zeros
      * edge-weighted average of neighbouring encoded electrode features
        (matching train_gnn.py's ElectrodeOnlyCNN_GNN)
      * nearest neighbour: copy the embedding of the truly closest electrode
        in 2D, via scipy.interpolate.NearestNDInterpolator
      * piecewise-linear interpolation of all electrode embeddings to each TM
        position via scipy.interpolate.LinearNDInterpolator (with a nearest-
        neighbour fallback for TM points outside the convex hull)
  - Two GATv2 layers (with edge_attr) propagate features across electrode<->TM edges
  - Linear classifier produces 2-class logits per node (positive class is
    fibrotic / higher_k / lower_a / excitable depending on the dataset)
  - Loss is computed on TM nodes only (electrodes have y=-100, ignored via
    the existing build_loss_fn from train_gnn.py)

Default loss is `dice_ce` (Dice + cross-entropy). Use `--loss-name ce` for plain CE.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.nn import Linear
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

from train_cnn_electrode import ElectrodeCNN
from train_gnn import LazyGraphDataset, build_loss_fn, set_seed

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


# ----------------------------- model -----------------------------

class CNN_GAT_Classifier(nn.Module):
    """Frozen (or fine-tunable) CNN encoder feeding two GATv2 layers; predicts on TM nodes.

    Args:
        cnn: a pretrained ElectrodeCNN (its forward must accept (B, T) and
            return (logits, embedding) when called with return_embedding=True).
        embed_dim: dimensionality of the CNN embedding (= input dim of GAT1).
        gnn_ch1, gnn_ch2: hidden dim of the two GAT layers.
        edge_dim: edge feature dim (1 for the 1/distance scalar weights).
        tm_feature_mode:
            'zeros': TM node features start at zeros.
            'weighted_elec_sum': TM node features are an edge-weighted average of
                neighbouring electrode embeddings (uses edge_attr if present, else
                uniform).
            'nearest_elec': TM node features are copied from the truly nearest
                electrode in 2D, via scipy.interpolate.NearestNDInterpolator
                fit on the electrode coordinates. Per-graph, runs on CPU/numpy.
            'interp_elec': TM node features are interpolated from ALL electrode
                embeddings via scipy's LinearNDInterpolator (piecewise-linear
                on the Delaunay triangulation of the electrode 2D positions).
                Any TM point outside the convex hull is filled from the nearest
                electrode. Per-graph, runs on CPU/numpy and is non-differentiable
                through the TM-init step (downstream gradients still flow via
                the electrode features themselves).
        n_classes: 2 for binary patch-vs-healthy (patch = fibrotic / higher_k /
            lower_a / excitable, depending on the dataset).
        freeze_cnn: if True, the CNN is held in eval() and its parameters are
            excluded from the optimizer (no fine-tuning).
    """

    def __init__(
        self,
        cnn,
        embed_dim,
        gnn_ch1=32,
        gnn_ch2=32,
        edge_dim=1,
        tm_feature_mode="interp_elec",
        n_classes=2,
        freeze_cnn=True,
        num_gat_layers=2,
        classifier_head="new",
        skip_connection=False,
    ):
        super().__init__()
        if tm_feature_mode not in {"zeros", "weighted_elec_sum", "nearest_elec", "interp_elec"}:
            raise ValueError(f"Unknown tm_feature_mode: {tm_feature_mode}")
        if num_gat_layers not in (0, 1, 2):
            raise ValueError(f"num_gat_layers must be 0, 1 or 2; got {num_gat_layers}")
        if classifier_head not in {"new", "cnn"}:
            raise ValueError(f"classifier_head must be 'new' or 'cnn'; got {classifier_head}")
        self.cnn = cnn
        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad_(False)
        self.embed_dim = embed_dim
        self.tm_feature_mode = tm_feature_mode
        self.num_gat_layers = num_gat_layers
        self.classifier_head = classifier_head
        # `skip` is only meaningful when there is a GAT stack to bypass.
        self.skip_connection = bool(skip_connection) and num_gat_layers > 0

        # Build 0, 1 or 2 GATv2 layers. With 0 layers the CNN-embedding / TM-init
        # flows straight into the classifier (no message passing -> no smoothing).
        self.gats = nn.ModuleList()
        in_dim = embed_dim
        out_dim = embed_dim
        for li in range(num_gat_layers):
            out_dim = gnn_ch1 if li == 0 else gnn_ch2
            self.gats.append(GATv2Conv(in_dim, out_dim, edge_dim=edge_dim))
            in_dim = out_dim
        last_dim = out_dim if num_gat_layers > 0 else embed_dim
        clf_in = last_dim + (embed_dim if self.skip_connection else 0)

        if classifier_head == "cnn":
            # Reuse the CNN's own trained classifier; it expects an embed_dim input,
            # so this is only valid as a pure pass-through (no GAT, no skip). This
            # reproduces the nearest-electrode baseline when tm_feature_mode='nearest_elec'.
            if num_gat_layers != 0 or self.skip_connection:
                raise ValueError(
                    "classifier_head='cnn' needs num_gat_layers=0 and no skip "
                    "(the CNN classifier expects an embed_dim input)."
                )
            if not hasattr(self.cnn, "classifier"):
                raise ValueError("Pretrained CNN has no `.classifier` to reuse.")
            self.classifier = None  # use self.cnn.classifier in forward
        else:
            self.classifier = Linear(clf_in, n_classes)

    def train(self, mode=True):
        # Keep the frozen CNN in eval mode regardless of parent state — so dropout
        # and any norm layers behave deterministically during downstream training.
        super().train(mode)
        if self.freeze_cnn:
            self.cnn.eval()
        return self

    def _encode_electrodes(self, x_elec):
        if self.freeze_cnn:
            with torch.no_grad():
                _, emb = self.cnn(x_elec, return_embedding=True)
            return emb
        _, emb = self.cnn(x_elec, return_embedding=True)
        return emb

    def _init_tm_weighted(self, x_node, edge_index, edge_attr, elec_mask, tm_mask):
        """Set TM rows of x_node to the edge-weighted average of neighbour electrodes.

        Mirrors train_gnn.ElectrodeOnlyCNN_GNN._init_tm_from_weighted_electrode_sum so
        results stay comparable between the two training scripts.
        """
        src_all, dst_all = edge_index[0], edge_index[1]
        m_fwd = elec_mask[src_all] & tm_mask[dst_all]
        m_rev = elec_mask[dst_all] & tm_mask[src_all]
        if not (m_fwd.any() or m_rev.any()):
            return x_node
        src_idx = torch.cat([src_all[m_fwd], dst_all[m_rev]], dim=0)
        dst_idx = torch.cat([dst_all[m_fwd], src_all[m_rev]], dim=0)
        if edge_attr is None:
            weights = x_node.new_ones(src_idx.size(0))
        else:
            w = torch.cat([edge_attr[m_fwd], edge_attr[m_rev]], dim=0)
            if w.dim() > 1:
                w = w[:, 0]
            weights = w.reshape(-1).abs()
        denom = x_node.new_zeros(x_node.size(0))
        denom.index_add_(0, dst_idx, weights)
        norm_w = weights / (denom[dst_idx] + 1e-8)
        tm_updates = x_node[src_idx] * norm_w.unsqueeze(1)
        x_node.index_add_(0, dst_idx, tm_updates)
        return x_node

    def _iter_graph_grids(self, x_node, elec_mask, tm_mask, elec_coords, batch_idx):
        """Yield ``(elec_xy, tm_xy, tm_node_indices, elec_feats)`` per graph in
        the batch — the shared boilerplate for any coord-based TM-init mode.

        TM positions are reconstructed as a square grid spanning the electrode
        bounding box, mirroring ``preprocess_graphs.py`` / ``train_gnn.py``.
        ``elec_feats`` is detached from autograd (interpolation runs in numpy).
        """
        if isinstance(elec_coords, (list, tuple)):
            coords_list = [np.asarray(c, dtype=np.float64) for c in elec_coords]
        elif elec_coords is not None:
            coords_list = [np.asarray(elec_coords, dtype=np.float64)]
        else:
            raise ValueError("coord-based TM init requires graph.elec_coords to be set.")

        num_graphs = 1 if batch_idx is None else int(batch_idx.max().item()) + 1
        if len(coords_list) != num_graphs:
            raise ValueError(
                f"coord-based TM init: got elec_coords for {len(coords_list)} graphs "
                f"but batch contains {num_graphs}."
            )

        for g_id in range(num_graphs):
            if batch_idx is None:
                g_node = torch.ones(x_node.size(0), dtype=torch.bool, device=x_node.device)
            else:
                g_node = batch_idx == g_id
            g_elec = g_node & elec_mask
            g_tm = g_node & tm_mask
            if not g_elec.any() or not g_tm.any():
                continue

            ec = coords_list[g_id][:, :2]
            n_elec = int(g_elec.sum().item())
            if ec.shape[0] != n_elec:
                # Defensive: trust the per-graph electrode-node count over an
                # over-sized stored coord array (matches train_gnn rebuild).
                ec = ec[:n_elec]

            n_tm = int(g_tm.sum().item())
            side = int(round(n_tm ** 0.5))
            if side * side != n_tm:
                raise ValueError(
                    f"coord-based TM init: TM-node count {n_tm} is not a perfect "
                    "square; cannot infer grid positions."
                )
            # TM node k = (row=k//side, col=k%side), matching gt_label.reshape / y_nodes.
            # Columns are (axis0=row, axis1=col) to align with ec, so the nearest-electrode
            # assignment is physical. NB: the OLD code used meshgrid(xg, yg) +
            # column_stack([XX.ravel(), YY.ravel()]), which places node k at (xg[k%side],
            # yg[k//side]) -- a row<->col transpose vs the GT (verified by check_tm_assignment.py:
            # s1 centroids were the exact transpose of GT). Use indexing="ij" so node k =
            # (g0[k//side], g1[k%side]).
            r_min, r_max = float(ec[:, 0].min()), float(ec[:, 0].max())  # axis0 (row)
            c_min, c_max = float(ec[:, 1].min()), float(ec[:, 1].max())  # axis1 (col)
            g0 = np.linspace(r_min, r_max, side)
            g1 = np.linspace(c_min, c_max, side)
            GG0, GG1 = np.meshgrid(g0, g1, indexing="ij")
            tm_xy = np.column_stack([GG0.ravel(), GG1.ravel()])

            e_feats = x_node[g_elec].detach().cpu().numpy().astype(np.float64)
            tm_idx = g_tm.nonzero(as_tuple=False).squeeze(-1)
            yield ec, tm_xy, tm_idx, e_feats

    def _init_tm_nearest(self, x_node, elec_mask, tm_mask, elec_coords, batch_idx):
        """Set TM rows of x_node to the embedding of the truly nearest electrode
        in 2D, via ``scipy.interpolate.NearestNDInterpolator``.
        """
        from scipy.interpolate import NearestNDInterpolator
        for ec, tm_xy, tm_idx, e_feats in self._iter_graph_grids(
            x_node, elec_mask, tm_mask, elec_coords, batch_idx
        ):
            nn = NearestNDInterpolator(ec, e_feats)
            t_feats = nn(tm_xy)
            x_node[tm_idx] = torch.as_tensor(
                t_feats, device=x_node.device, dtype=x_node.dtype
            )
        return x_node

    def _init_tm_interp(self, x_node, elec_mask, tm_mask, elec_coords, batch_idx):
        """Set TM rows of x_node to a piecewise-linear interpolation of all
        electrode embeddings via ``scipy.interpolate.LinearNDInterpolator``
        (barycentric on the Delaunay triangulation). TM points outside the
        convex hull are backfilled from ``NearestNDInterpolator``.
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        for ec, tm_xy, tm_idx, e_feats in self._iter_graph_grids(
            x_node, elec_mask, tm_mask, elec_coords, batch_idx
        ):
            lin = LinearNDInterpolator(ec, e_feats)
            t_feats = lin(tm_xy)
            nan_rows = np.isnan(t_feats).any(axis=1)
            if nan_rows.any():
                nn = NearestNDInterpolator(ec, e_feats)
                t_feats[nan_rows] = nn(tm_xy[nan_rows])
            x_node[tm_idx] = torch.as_tensor(
                t_feats, device=x_node.device, dtype=x_node.dtype
            )
        return x_node

    def forward(self, x, edge_index, edge_attr=None, tm_mask=None,
                batch_idx=None, elec_coords=None):
        if tm_mask is None:
            raise ValueError("tm_mask is required.")
        tm_mask = tm_mask.bool()
        elec_mask = ~tm_mask
        n_nodes = x.shape[0]
        x_node = x.new_zeros((n_nodes, self.embed_dim))
        if elec_mask.any():
            x_node[elec_mask] = self._encode_electrodes(x[elec_mask])
        if tm_mask.any():
            if self.tm_feature_mode == "weighted_elec_sum":
                x_node = self._init_tm_weighted(x_node, edge_index, edge_attr, elec_mask, tm_mask)
            elif self.tm_feature_mode == "nearest_elec":
                x_node = self._init_tm_nearest(x_node, elec_mask, tm_mask, elec_coords, batch_idx)
            elif self.tm_feature_mode == "interp_elec":
                x_node = self._init_tm_interp(x_node, elec_mask, tm_mask, elec_coords, batch_idx)
        h = x_node
        for gat in self.gats:
            h = gat(h, edge_index, edge_attr)
            h = F.relu(h)
        # Optional skip: let the un-smoothed per-node embedding bypass the GAT stack.
        feat = torch.cat([h, x_node], dim=1) if self.skip_connection else h
        if self.classifier_head == "cnn":
            logits = self.cnn.classifier(feat)
        else:
            logits = self.classifier(feat)
        return logits, feat


# ----------------------------- io helpers -----------------------------

def load_pretrained_cnn(ckpt_path, device):
    """Load a pretrained ElectrodeCNN. Returns (cnn, cfg, embed_dim).

    ``embed_dim`` is discovered from a dummy forward pass rather than read from
    ``cfg`` because the current ElectrodeCNN exposes its embedding via a
    LazyLinear(conv_ch) head and no longer stores ``embed_dim`` in its config.
    The dummy pass also materialises the LazyLinear so ``load_state_dict`` can
    succeed (LazyLinear weights must exist before being loaded into).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_config" not in ckpt:
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no 'model_config'. "
            "Was it saved by train_cnn_electrode.py?"
        )
    cfg = ckpt["model_config"]
    cnn = ElectrodeCNN(**cfg).to(device)
    T = int(cfg.get("T", 1500))
    with torch.no_grad():
        _, emb = cnn(torch.zeros(1, T, device=device), return_embedding=True)
    embed_dim = int(emb.shape[-1])
    cnn.load_state_dict(ckpt["model_state_dict"])
    cnn.eval()
    return cnn, cfg, embed_dim


def run_one_pass(model, loader, device, loss_fn, component_fn, component_names,
                 train_mode, optimizer=None):
    """Run one epoch (train or eval). Returns (avg_loss, components_dict, preds, labels, probs).

    ``probs`` is the softmax probability of the positive class (label==1) on valid
    TM nodes — needed for threshold-independent metrics (ROC AUC, PR AUC).
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    total = 0.0
    n_batches = 0
    comp_running = {n: 0.0 for n in (component_names or [])}
    all_preds, all_labels, all_probs = [], [], []

    cm = torch.enable_grad() if train_mode else torch.no_grad()
    with cm:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            edge_attr = getattr(batch, "edge_attr", None)
            if train_mode:
                optimizer.zero_grad()
            ec = getattr(batch, "elec_coords", None)
            # If PyG concatenated per-graph numpy arrays into one block, split
            # back using num_elec (per-graph electrode count).
            if ec is not None and not isinstance(ec, (list, tuple)):
                ne = getattr(batch, "num_elec", None)
                if ne is not None and (torch.is_tensor(ne) or hasattr(ne, "__len__")):
                    ne_list = ne.tolist() if torch.is_tensor(ne) else list(ne)
                    if len(ne_list) > 1:
                        ec_np = ec.detach().cpu().numpy() if torch.is_tensor(ec) else np.asarray(ec)
                        chunks, start = [], 0
                        for n in ne_list:
                            chunks.append(ec_np[start:start + int(n)])
                            start += int(n)
                        ec = chunks
            logits, _ = model(batch.x, batch.edge_index, edge_attr,
                              tm_mask=batch.tm_mask,
                              batch_idx=getattr(batch, "batch", None),
                              elec_coords=ec)
            loss = loss_fn(logits, batch.y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            n_batches += 1
            if component_fn is not None:
                comps = component_fn(logits, batch.y)
                for name, v in zip(component_names, comps):
                    comp_running[name] += float(v.item())
            tm = batch.tm_mask.bool()
            tm_logits = logits[tm]
            preds = tm_logits.argmax(dim=1)
            probs = F.softmax(tm_logits, dim=1)[:, 1]
            labels = batch.y[tm]
            valid = labels != -100
            all_preds.append(preds[valid].detach().cpu().numpy())
            all_labels.append(labels[valid].detach().cpu().numpy())
            all_probs.append(probs[valid].detach().cpu().numpy())

    avg = total / max(1, n_batches)
    comps_avg = {k: v / max(1, n_batches) for k, v in comp_running.items()}
    p_all = np.concatenate(all_preds) if all_preds else np.empty(0, dtype=np.int64)
    l_all = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    pr_all = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float32)
    return avg, comps_avg, p_all, l_all, pr_all


def safe_roc_auc(labels, probs):
    """ROC AUC with a NaN fallback when only one class is present."""
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def safe_average_precision(labels, probs):
    """Average precision (PR AUC) with a NaN fallback when only one class is present."""
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def plot_roc(labels, probs, out_path, auc=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig = plt.figure(figsize=(5, 5))
    label = f"AUC = {auc:.3f}" if auc is not None and not np.isnan(auc) else None
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    if label is not None:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_pr(labels, probs, out_path, ap=None):
    precision, recall, _ = precision_recall_curve(labels, probs)
    fig = plt.figure(figsize=(5, 5))
    label = f"AP = {ap:.3f}" if ap is not None and not np.isnan(ap) else None
    plt.plot(recall, precision, label=label)
    pos_rate = float((labels == 1).mean()) if labels.size else 0.0
    plt.hlines(pos_rate, 0, 1, linestyles="--", colors="gray", linewidth=0.8,
               label=f"baseline = {pos_rate:.3f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ----------------------------- TM grid plotting -----------------------------

def plot_tm_grid_overlay(graph, preds, labels, out_path, target_shape=(30, 30)):
    """Single-panel TM-node overlay: prediction filled, ground truth as a contour."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_tm = labels.size
    if n_tm != target_shape[0] * target_shape[1]:
        side = int(round(n_tm ** 0.5))
        target_shape = (side, side)

    gt = labels.reshape(target_shape)
    pr = preds.reshape(target_shape)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    # Prediction as the filled base image.
    ax.imshow(pr, origin="lower", cmap="Blues", vmin=0, vmax=1)
    # Ground truth fibrosis boundary as a contour on top (only if both classes present).
    if gt.min() < 0.5 < gt.max():
        ax.contour(gt, levels=[0.5], colors="red", linewidths=1.5, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Prediction (fill) + ground truth (contour)")

    # Overlays use the *full-domain (row, col)* frame the GT/pred image is built on:
    # gt[row, col] resized to target_shape, shown by imshow with row (axis 0) vertical
    # and col (axis 1) horizontal. So col -> scatter-x, row -> scatter-y, normalized
    # over the full ncells grid (not the electrode span).
    meta = getattr(graph, "meta", None) or {}
    ncells = meta.get("ncells", 100)
    ncells = int(np.asarray(ncells).ravel()[0]) if ncells is not None else 100
    denom = max(1, ncells - 1)

    legend = [
        Patch(facecolor=plt.cm.Blues(0.85), edgecolor="none", label="Prediction"),
        Line2D([], [], color="red", lw=1.5, label="Ground truth"),
    ]

    elec_coords = getattr(graph, "elec_coords", None)
    if elec_coords is not None:
        ec = np.asarray(elec_coords, dtype=float)
        # elec_coords were saved as raw cell-grid positions / 10 (see load_sim_file),
        # and are ordered (axis0, axis1) like stim_centers. Recover cell units (* 10),
        # then map col -> x and row -> y on the full domain so dots share the cross frame.
        ec_cell = ec * 10.0
        xs = ec_cell[:, 1] / denom * (target_shape[1] - 1)  # col (axis 1) -> horizontal
        ys = ec_cell[:, 0] / denom * (target_shape[0] - 1)  # row (axis 0) -> vertical
        ax.scatter(xs, ys, s=8, c="white", edgecolors="black", linewidths=0.4)
        legend.append(Line2D([], [], marker="o", linestyle="none", markersize=5,
                             markerfacecolor="white", markeredgecolor="black",
                             label="Electrodes"))

    # Stimulus sites: stim_centers are (row, col) integer indices on the same grid.
    stim = meta.get("stim_centers", meta.get("stim_center", None))
    if stim is not None:
        stim = np.asarray(stim, dtype=float).reshape(-1, 2)
        if stim.size:
            sx = stim[:, 1] / denom * (target_shape[1] - 1)  # col -> horizontal
            sy = stim[:, 0] / denom * (target_shape[0] - 1)  # row -> vertical
            ax.scatter(sx, sy, s=80, marker="X", c="cyan",
                       edgecolors="black", linewidths=0.8, zorder=5)
            legend.append(Line2D([], [], marker="X", linestyle="none", markersize=9,
                                 markerfacecolor="cyan", markeredgecolor="black",
                                 label="Stimuli"))

    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_indexed_overlays(graphs_dir, indices, model, device, out_dir):
    """Plot specific graphs chosen by simulation index (stem ends in '_<index>').

    Loads the matching .pt graphs directly from the graphs-dir index (regardless of the
    train/val split) so the same sims can be compared against the extrapolate_cnn_to_tm.py
    baseline plots. Returns the list of written paths.
    """
    out_dir = Path(out_dir) / "indexed_overlays"
    out_dir.mkdir(exist_ok=True)
    index_file = Path(graphs_dir) / "graph_index.txt"
    paths = [ln.strip() for ln in index_file.read_text().splitlines() if ln.strip()]

    written = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            matches = [p for p in paths if Path(p).stem.endswith(f"_{idx}")]
            if not matches:
                print(f"--plot-indices: no graph matching '_{idx}' in {index_file}")
                continue
            gp = matches[0]
            gp = gp if Path(gp).is_absolute() else str(Path(graphs_dir) / Path(gp).name)
            graph = torch.load(gp, weights_only=False, map_location="cpu")
            g = graph.clone().to(device)
            edge_attr = getattr(g, "edge_attr", None)
            logits, _ = model(g.x, g.edge_index, edge_attr, tm_mask=g.tm_mask,
                              elec_coords=getattr(g, "elec_coords", None))
            tm = g.tm_mask.bool()
            preds = logits[tm].argmax(dim=1).cpu().numpy()
            labels = g.y[tm].cpu().numpy()
            stem = Path(getattr(graph, "pacing_info", gp)).stem
            out_path = out_dir / f"overlay_idx{idx}_{stem}.png"
            try:
                plot_tm_grid_overlay(graph, preds, labels, out_path)
                written.append(out_path)
            except Exception as e:
                print(f"Indexed overlay failed on idx {idx} ({stem}): {e}")
    return written


def save_random_val_overlays(val_set, model, device, out_dir, n=5, seed=0):
    out_dir = Path(out_dir) / "val_overlays"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    n_pick = min(n, len(val_set))
    chosen = rng.choice(len(val_set), size=n_pick, replace=False)
    written = []
    model.eval()
    with torch.no_grad():
        for i in chosen:
            graph = val_set[int(i)]
            g = graph.clone().to(device)
            edge_attr = getattr(g, "edge_attr", None)
            logits, _ = model(g.x, g.edge_index, edge_attr, tm_mask=g.tm_mask,
                              elec_coords=getattr(g, "elec_coords", None))
            tm = g.tm_mask.bool()
            preds = logits[tm].argmax(dim=1).cpu().numpy()
            labels = g.y[tm].cpu().numpy()
            stem = Path(getattr(graph, "pacing_info", f"graph_{int(i)}")).stem
            out_path = out_dir / f"overlay_{stem}.png"
            try:
                plot_tm_grid_overlay(graph, preds, labels, out_path)
                written.append(out_path)
            except Exception as e:
                print(f"Overlay failed on {stem}: {e}")
    return written


# ----------------------------- main -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn-ckpt", required=True,
                   help="Path to best_model.pt produced by train_cnn_electrode.py.")
    p.add_argument("--graphs-dir", required=True,
                   help="Directory containing graph_index.txt + .pt graphs (from preprocess_graphs.py).")
    p.add_argument("--out-root", default="cnn_gat_runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-graphs", type=int, default=6000)
    p.add_argument("--rebuild-k", type=int, default=None,
                   help="If set, rebuild elec->TM edges on the fly (see train_gnn.LazyGraphDataset).")
    p.add_argument("--tm-feature-mode",
                   choices=["zeros", "weighted_elec_sum", "nearest_elec", "interp_elec"],
                   default="interp_elec")
    p.add_argument("--freeze-cnn", action="store_true")
    p.add_argument("--no-freeze-cnn", dest="freeze_cnn", action="store_false")
    p.set_defaults(freeze_cnn=True)
    p.add_argument("--gnn-ch1", type=int, default=32)
    p.add_argument("--gnn-ch2", type=int, default=32)
    p.add_argument("--num-gat-layers", type=int, default=2, choices=[0, 1, 2],
                   help="Number of GATv2 layers. 0 = CNN-embedding / TM-init flows straight "
                        "to the classifier (no message passing, no smoothing).")
    p.add_argument("--classifier-head", choices=["new", "cnn"], default="new",
                   help="'new': train a fresh Linear head on the GAT output. 'cnn': reuse the "
                        "frozen CNN's own classifier (requires --num-gat-layers 0 and no skip); "
                        "with --tm-feature-mode nearest_elec this reproduces the baseline.")
    p.add_argument("--skip-connection", action="store_true",
                   help="Concatenate the original CNN-embedding / TM-init with the GAT output "
                        "before the classifier, so the crisp per-node signal bypasses smoothing.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--loss-name", default="dice_ce", choices=["ce", "dice", "dice_ce", "dice_bce"])
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--dice-smooth", type=float, default=1e-5)
    p.add_argument("--n-overlay-plots", type=int, default=5)
    p.add_argument("--overlay-seed", type=int, default=0)
    p.add_argument("--plot-indices", default=None,
                   help="Comma-separated simulation indices to plot (matches graph stems "
                        "ending in '_<index>'), e.g. '26,2383,3709'. Plots these exact graphs "
                        "(train or val) so the same sims can be compared against the "
                        "extrapolate_cnn_to_tm.py baseline. Saved under <out>/indexed_overlays/.")
    p.add_argument("--patience", type=int, default=50,
                   help="Stop training if val loss does not improve by --min-delta for this many "
                        "consecutive epochs. Set to 0 to disable early stopping.")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="Minimum decrease in val loss to count as an improvement.")
    p.add_argument("--positive-class-name", default="fibrotic",
                   help="Display name for label==1 in plots, reports, and wandb. "
                        "Set per task: 'fibrotic', 'higher_k', 'lower_a', 'excitable' (a&k both), etc.")
    p.add_argument("--negative-class-name", default="healthy",
                   help="Display name for label==0 (normal myocardium).")
    p.add_argument("--wandb-project", default="gnn_cardiac_fibrosis",
                   help="WandB project name (matches train_gnn.py).")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="WandB mode. Default: online if WANDB_API_KEY is set or "
                        "WANDB_MODE env var is set, else offline. Use 'disabled' to skip wandb.")
    return p.parse_args()


def init_wandb(args, run_name, extra_config):
    """Initialize wandb run, mirroring train_gnn.py's behaviour. Returns the run or None."""
    if not HAS_WANDB:
        print("wandb not installed; skipping experiment tracking.")
        return None
    mode = args.wandb_mode
    if mode is None:
        mode = os.getenv("WANDB_MODE")
    if mode == "disabled":
        return None
    if mode is None:
        if os.getenv("WANDB_API_KEY"):
            wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=False)
            mode = "online"
        else:
            mode = "offline"
    elif mode == "online" and os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=False)

    return wandb.init(
        project=args.wandb_project,
        name=str(run_name),
        mode=mode,
        config={**vars(args), **extra_config},
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    cnn, cnn_cfg, embed_dim = load_pretrained_cnn(args.cnn_ckpt, device)
    print(f"Loaded CNN: embed_dim={embed_dim}, "
          f"params={sum(p.numel() for p in cnn.parameters())}")

    index_file = Path(args.graphs_dir) / "graph_index.txt"
    if not index_file.exists():
        raise FileNotFoundError(f"Missing {index_file}. Run preprocess_graphs.py first.")
    dataset = LazyGraphDataset(
        index_file, max_graphs=args.max_graphs,
        base_dir=args.graphs_dir, rebuild_k=args.rebuild_k,
    )
    print(f"Loaded {len(dataset)} graphs from {args.graphs_dir}")

    n_total = len(dataset)
    n_val = max(1, int(round(args.val_frac * n_total)))
    n_train = n_total - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        exclude_keys=["pacing_info", "meta"],
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        exclude_keys=["pacing_info", "meta"],
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = CNN_GAT_Classifier(
        cnn=cnn,
        embed_dim=embed_dim,
        gnn_ch1=args.gnn_ch1,
        gnn_ch2=args.gnn_ch2,
        tm_feature_mode=args.tm_feature_mode,
        freeze_cnn=args.freeze_cnn,
        n_classes=2,
        num_gat_layers=args.num_gat_layers,
        classifier_head=args.classifier_head,
        skip_connection=args.skip_connection,
    ).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # No trainable params (e.g. 0 GAT layers + reused CNN classifier, frozen CNN) ->
    # nothing to optimize, so just evaluate the fixed-weight model (baseline check).
    eval_only = n_train_params == 0
    print(f"GNN trainable params: {n_train_params}"
          + ("  -> EVAL-ONLY run (fixed weights, reproduces the baseline)" if eval_only else ""))

    optimizer = None
    if not eval_only:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    loss_fn, component_fn, component_names = build_loss_fn(
        args.loss_name, num_classes=2, ignore_index=-100,
        dice_smooth=args.dice_smooth, dice_weight=args.dice_weight,
        ce_weight=args.ce_weight, bce_weight=args.bce_weight,
    )

    run_name = args.run_name or (
        f"cnn_{Path(args.cnn_ckpt).parent.name}_tm-{args.tm_feature_mode}"
        f"__{Path(args.graphs_dir).name}_{args.loss_name}"
    )
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {**vars(args), "embed_dim": embed_dim, "device": str(device),
             "n_train": len(train_set), "n_val": len(val_set),
             "n_train_params": n_train_params},
            f, indent=2,
        )

    wandb_run = init_wandb(args, run_name, extra_config={
        "embed_dim": embed_dim,
        "device": str(device),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_train_params": n_train_params,
        "out_dir": str(out_dir),
    })

    train_losses, val_losses = [], []
    train_components = {n: [] for n in (component_names or [])}
    val_components = {n: [] for n in (component_names or [])}
    best_val = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if eval_only:
            break
        last_epoch = epoch
        train_loss, train_comps, _, _, _ = run_one_pass(
            model, train_loader, device, loss_fn, component_fn, component_names,
            train_mode=True, optimizer=optimizer,
        )
        val_loss, val_comps, _, val_labels_ep, val_probs_ep = run_one_pass(
            model, val_loader, device, loss_fn, component_fn, component_names,
            train_mode=False,
        )
        val_roc_auc = safe_roc_auc(val_labels_ep, val_probs_ep)
        val_ap = safe_average_precision(val_labels_ep, val_probs_ep)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for k in train_components:
            train_components[k].append(train_comps[k])
            val_components[k].append(val_comps[k])

        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "embed_dim": embed_dim,
                    "gnn_ch1": args.gnn_ch1,
                    "gnn_ch2": args.gnn_ch2,
                    "tm_feature_mode": args.tm_feature_mode,
                    "freeze_cnn": args.freeze_cnn,
                    "num_gat_layers": args.num_gat_layers,
                    "classifier_head": args.classifier_head,
                    "skip_connection": args.skip_connection,
                },
                "cnn_ckpt": str(args.cnn_ckpt),
                "cnn_config": cnn_cfg,
            }, out_dir / "best_model.pt")
        else:
            epochs_since_improve += 1

        print(
            f"Epoch {epoch:04d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"val ROC-AUC {val_roc_auc:.4f} | val AP {val_ap:.4f} | "
            f"best {best_val:.4f} @ ep{best_epoch} | no-improve {epochs_since_improve}/{args.patience}"
        )

        if wandb_run is not None:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": val_roc_auc,
                "val_average_precision": val_ap,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "epochs_since_improve": epochs_since_improve,
            }
            for k, v in train_comps.items():
                log_dict[f"train_{k}"] = v
            for k, v in val_comps.items():
                log_dict[f"val_{k}"] = v
            wandb_run.log(log_dict)

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: no val-loss improvement "
                f"(>= {args.min_delta}) for {args.patience} consecutive epochs. "
                f"Best val {best_val:.4f} at epoch {best_epoch}."
            )
            break

    torch.save({"epoch": last_epoch, "model_state_dict": model.state_dict()},
               out_dir / "final_model.pt")

    # --- Loss plots ---
    plt.figure()
    plt.semilogy(train_losses, label="train")
    plt.semilogy(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(f"{args.loss_name} loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "losses.png")
    plt.close()

    if component_names:
        plt.figure()
        for name in component_names:
            plt.semilogy(train_components[name], label=f"train {name}")
            plt.semilogy(val_components[name], label=f"val {name}")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title(f"{args.loss_name} components")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "losses_components.png")
        plt.close()

    # Eval-only runs never enter the training loop, so no best_model.pt was written;
    # persist the fixed-weight model now so the final-eval reload below succeeds.
    if eval_only or not (out_dir / "best_model.pt").exists():
        torch.save({
            "epoch": last_epoch,
            "val_loss": None,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embed_dim": embed_dim,
                "gnn_ch1": args.gnn_ch1,
                "gnn_ch2": args.gnn_ch2,
                "tm_feature_mode": args.tm_feature_mode,
                "freeze_cnn": args.freeze_cnn,
                "num_gat_layers": args.num_gat_layers,
                "classifier_head": args.classifier_head,
                "skip_connection": args.skip_connection,
            },
            "cnn_ckpt": str(args.cnn_ckpt),
            "cnn_config": cnn_cfg,
        }, out_dir / "best_model.pt")

    # --- Final eval with best ckpt ---
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, _, val_preds, val_labels, val_probs = run_one_pass(
        model, val_loader, device, loss_fn, component_fn, component_names,
        train_mode=False,
    )
    val_roc_auc = safe_roc_auc(val_labels, val_probs)
    val_ap = safe_average_precision(val_labels, val_probs)
    print(
        f"Best val loss: {val_loss:.4f} | ROC-AUC: {val_roc_auc:.4f} | AP: {val_ap:.4f}"
    )

    plot_roc(val_labels, val_probs, out_dir / "roc_curve.png", auc=val_roc_auc)
    plot_pr(val_labels, val_probs, out_dir / "pr_curve.png", ap=val_ap)

    neg_name = args.negative_class_name
    pos_name = args.positive_class_name
    class_names = [neg_name, pos_name]

    report_dict = classification_report(
        val_labels, val_preds, labels=[0, 1],
        target_names=class_names, zero_division=0, output_dict=True,
    )
    report = classification_report(
        val_labels, val_preds, labels=[0, 1],
        target_names=class_names, zero_division=0,
    )
    print(report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Validation TM-node confusion matrix ({neg_name} vs {pos_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    np.savez(out_dir / "val_predictions.npz",
             preds=val_preds, labels=val_labels, probs=val_probs)

    if args.n_overlay_plots > 0:
        written = save_random_val_overlays(
            val_set, model, device, out_dir,
            n=args.n_overlay_plots, seed=args.overlay_seed,
        )
        print(f"Wrote {len(written)} overlay plot(s) to {out_dir / 'val_overlays'}")

    if args.plot_indices:
        idxs = [s.strip() for s in args.plot_indices.split(",") if s.strip()]
        written = save_indexed_overlays(args.graphs_dir, idxs, model, device, out_dir)
        print(f"Wrote {len(written)} indexed overlay plot(s) to {out_dir / 'indexed_overlays'}")

    if wandb_run is not None:
        final_metrics = {
            "final/best_val_loss": best_val,
            "final/best_epoch": best_epoch,
            "final/last_epoch": last_epoch,
            "final/accuracy": report_dict["accuracy"],
            f"final/{neg_name}_precision": report_dict[neg_name]["precision"],
            f"final/{neg_name}_recall": report_dict[neg_name]["recall"],
            f"final/{neg_name}_f1": report_dict[neg_name]["f1-score"],
            f"final/{pos_name}_precision": report_dict[pos_name]["precision"],
            f"final/{pos_name}_recall": report_dict[pos_name]["recall"],
            f"final/{pos_name}_f1": report_dict[pos_name]["f1-score"],
            "final/macro_f1": report_dict["macro avg"]["f1-score"],
            "final/roc_auc": val_roc_auc,
            "final/average_precision": val_ap,
        }
        # Attach the rendered figures.
        loss_png = out_dir / "losses.png"
        cm_png = out_dir / "confusion_matrix.png"
        roc_png = out_dir / "roc_curve.png"
        pr_png = out_dir / "pr_curve.png"
        if loss_png.exists():
            final_metrics["final/losses"] = wandb.Image(str(loss_png))
        if cm_png.exists():
            final_metrics["final/confusion_matrix"] = wandb.Image(str(cm_png))
        if roc_png.exists():
            final_metrics["final/roc_curve"] = wandb.Image(str(roc_png))
        if pr_png.exists():
            final_metrics["final/pr_curve"] = wandb.Image(str(pr_png))
        # Also send a wandb-native confusion matrix so it is interactive in the UI.
        final_metrics["final/confusion_matrix_table"] = wandb.plot.confusion_matrix(
            preds=val_preds.tolist(), y_true=val_labels.tolist(),
            class_names=class_names,
        )
        # Interactive ROC / PR curves (positive class index 1).
        # wandb expects per-class score columns, so stack [1-p, p].
        probs_2col = np.stack([1.0 - val_probs, val_probs], axis=1)
        if val_labels.size and np.unique(val_labels).size >= 2:
            final_metrics["final/roc_curve_table"] = wandb.plot.roc_curve(
                y_true=val_labels, y_probas=probs_2col,
                labels=class_names,
            )
            final_metrics["final/pr_curve_table"] = wandb.plot.pr_curve(
                y_true=val_labels, y_probas=probs_2col,
                labels=class_names,
            )
        wandb_run.log(final_metrics)
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["roc_auc"] = val_roc_auc
        wandb_run.summary["average_precision"] = val_ap
        wandb_run.finish()

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
