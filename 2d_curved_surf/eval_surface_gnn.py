"""
eval_surface_gnn.py

Apply a CNN+GAT checkpoint trained on the flat 2D grid
(train_gnn_with_pretrained_cnn.py) to curved-surface graphs built by
surface_to_graph.py -- and see whether fibrosis detection transfers.

The only part of the trained model that is NOT geometry-agnostic is the TM-init
step: for tm_feature_mode in {interp_elec, nearest_elec} the original model
reconstructs a square TM grid from the electrode bounding box and ignores the
real node coordinates (fine for a 30x30 grid, wrong for a scattered surface).
``SurfaceCNN_GAT`` overrides that step to interpolate electrode embeddings onto
the REAL per-node surface parameters stored on each graph as ``tm_coords``
(theta, phi). Everything downstream (the frozen CNN encoder, both GATv2 layers,
the classifier) is reused unchanged, so the transfer is faithful to the trained
weights.

Usage:
    python eval_surface_gnn.py \
        --gat-ckpt /Users/mac/Desktop/Imperial PhD/GNN/pretrained_elec_cnn_gat_V_linear_interp_elec_nofreeze/best_model.pt \
        --graphs-dir surface_graphs/ \
        --out-dir surface_eval
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from train_cnn_electrode import ElectrodeCNN
from train_gnn_with_pretrained_cnn import CNN_GAT_Classifier


# --------------------------------------------------------------- model

class LegacyElectrodeCNN(ElectrodeCNN):
    """ElectrodeCNN as it was when the older (16-d embedding) checkpoints --
    e.g. pretrained_elec_cnn_gat_V_linear_interp_elec[_nofreeze] -- were trained.

    The conv `features` stack is unchanged, but the embedding head was a single
    Linear (`embed`, no LayerNorm) producing an explicit ``embed_dim`` (16),
    whereas today's ElectrodeCNN uses ``head`` = LazyLinear(conv_ch) + LayerNorm.
    That old class is no longer in the repo (train_cnn_electrode.py is untracked
    and was overwritten), so we reconstruct it here to match the checkpoint's
    state_dict: keys ``cnn.embed.1.*`` and ``classifier`` = Linear(embed_dim, 2).

    NB: the parameter-bearing layers are pinned by the checkpoint (a strict load
    verifies them). The only un-pinnable detail is the parameter-free activation
    after the Linear; the original used ReLU (standard for this encoder).
    """

    def __init__(self, embed_dim=16, n_classes=2, **kw):
        super().__init__(n_classes=n_classes, **kw)   # builds features/head/classifier
        del self.head                                 # drop the modern head
        # The checkpoint's embed Linear has a conv_ch-sized input (weight
        # [16,16]), not the flattened time axis -- so the old model pooled over
        # time before the Linear. Append a global temporal pool (param-free, so
        # the features.* keys still match) to collapse (N, conv_ch, T') -> (N,
        # conv_ch, 1) -> Flatten -> conv_ch.
        self.features = nn.Sequential(*self.features, nn.AdaptiveAvgPool1d(1))
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(embed_dim),                 # -> cnn.embed.1.* (conv_ch -> embed_dim)
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_embedding=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        f = self.features(x)
        emb = self.embed(f)
        out = self.classifier(emb)
        return (out, emb) if return_embedding else out

class SurfaceCNN_GAT(CNN_GAT_Classifier):
    """CNN_GAT_Classifier whose coord-based TM-init uses the REAL surface node
    parameters (graph.tm_coords) instead of a reconstructed square grid.

    forward() here handles a single graph at a time (the eval loop below feeds
    one graph per call), which keeps the elec_coords / tm_coords mapping
    unambiguous on a scattered surface.
    """

    def _interp_to_real_coords(self, elec_feats, elec_xy, tm_xy):
        """Piecewise-linear interpolation (interp_elec) or nearest (nearest_elec)
        of electrode embeddings onto the real TM coordinates."""
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        ec = np.asarray(elec_xy, dtype=np.float64)[:, :2]
        tc = np.asarray(tm_xy, dtype=np.float64)[:, :2]
        ef = np.asarray(elec_feats, dtype=np.float64)

        if self.tm_feature_mode == "nearest_elec":
            return NearestNDInterpolator(ec, ef)(tc)

        lin = LinearNDInterpolator(ec, ef)
        out = lin(tc)
        nan_rows = np.isnan(out).any(axis=1)
        if nan_rows.any():                      # outside convex hull -> nearest
            out[nan_rows] = NearestNDInterpolator(ec, ef)(tc[nan_rows])
        return out

    def forward(self, x, edge_index, edge_attr=None, tm_mask=None,
                elec_coords=None, tm_coords=None):
        if tm_mask is None:
            raise ValueError("tm_mask is required.")
        tm_mask = tm_mask.bool()
        elec_mask = ~tm_mask
        x_node = x.new_zeros((x.shape[0], self.embed_dim))
        if elec_mask.any():
            x_node[elec_mask] = self._encode_electrodes(x[elec_mask])

        if tm_mask.any():
            mode = self.tm_feature_mode
            if mode == "zeros":
                pass
            elif mode == "weighted_elec_sum":
                # Pure edge-based init -- geometry-agnostic, needs no coords.
                x_node = self._init_tm_weighted(x_node, edge_index, edge_attr,
                                                elec_mask, tm_mask)
            elif mode in ("interp_elec", "nearest_elec"):
                if elec_coords is None or tm_coords is None:
                    raise ValueError(
                        f"{mode} needs graph.elec_coords and graph.tm_coords "
                        "(built by surface_to_graph.py)."
                    )
                e_feats = x_node[elec_mask].detach().cpu().numpy()
                t_feats = self._interp_to_real_coords(e_feats, elec_coords, tm_coords)
                x_node[tm_mask] = torch.as_tensor(
                    t_feats, device=x_node.device, dtype=x_node.dtype)
            else:
                raise ValueError(f"Unknown tm_feature_mode: {mode}")

        h = x_node
        for gat in self.gats:
            h = F.relu(gat(h, edge_index, edge_attr))
        feat = torch.cat([h, x_node], dim=1) if self.skip_connection else h
        if self.classifier_head == "cnn":
            logits = self.cnn.classifier(feat)
        else:
            logits = self.classifier(feat)
        return logits, feat


def load_surface_model(gat_ckpt, device):
    """Rebuild SurfaceCNN_GAT from a train_gnn_with_pretrained_cnn checkpoint.

    The GAT checkpoint's model_state_dict already contains the (frozen) CNN
    submodule weights, so we materialise a fresh ElectrodeCNN + GAT and load the
    whole state dict -- no separate CNN ckpt needed.
    """
    ckpt = torch.load(gat_ckpt, map_location=device, weights_only=False)
    mcfg = ckpt["model_config"]
    sd = ckpt["model_state_dict"]
    cnn_cfg = dict(ckpt.get("cnn_config", {}))

    # Keep only kwargs the current ElectrodeCNN signature accepts, dropping legacy
    # ones (e.g. 'embed_dim', now derived at runtime / passed explicitly below).
    import inspect
    accepted = set(inspect.signature(ElectrodeCNN.__init__).parameters) - {"self"}
    cnn_cfg = {k: v for k, v in cnn_cfg.items() if k in accepted}

    embed_dim = int(mcfg["embed_dim"])
    # Pick the CNN variant that matches the checkpoint: older runs saved a
    # single-Linear `embed` head (keys cnn.embed.*); the current ElectrodeCNN
    # uses `head`=LazyLinear(conv_ch)+LayerNorm. Auto-detect from the keys so
    # load_state_dict succeeds either way.
    legacy = any(k.startswith("cnn.embed.") for k in sd)
    if legacy:
        print(f"load_surface_model: checkpoint uses the legacy 'embed' CNN head "
              f"(embed_dim={embed_dim}); reconstructing LegacyElectrodeCNN.")
        cnn = LegacyElectrodeCNN(embed_dim=embed_dim, **cnn_cfg).to(device)
    else:
        cnn = ElectrodeCNN(**cnn_cfg).to(device)

    T = int(cnn_cfg.get("T", 1500))
    with torch.no_grad():                       # materialise LazyLinear in CNN
        cnn(torch.zeros(1, T, device=device), return_embedding=True)

    model = SurfaceCNN_GAT(
        cnn=cnn,
        embed_dim=embed_dim,
        gnn_ch1=int(mcfg.get("gnn_ch1", 32)),
        gnn_ch2=int(mcfg.get("gnn_ch2", 32)),
        tm_feature_mode=mcfg.get("tm_feature_mode", "interp_elec"),
        freeze_cnn=bool(mcfg.get("freeze_cnn", True)),
        n_classes=2,
        num_gat_layers=int(mcfg.get("num_gat_layers", 2)),
        classifier_head=mcfg.get("classifier_head", "new"),
        skip_connection=bool(mcfg.get("skip_connection", False)),
    ).to(device)
    # Compat: older checkpoints named the two GAT layers gat1/gat2; the current
    # CNN_GAT_Classifier holds them in a `gats` ModuleList (gats.0/gats.1).
    # Remap so pre-`gats` checkpoints (e.g. the zero-shot model) still load.
    if any(k.startswith(("gat1.", "gat2.")) for k in sd) and \
            any(k.startswith("gats.") for k in model.state_dict()):
        sd = {("gats.0." + k[5:] if k.startswith("gat1.") else
               "gats.1." + k[5:] if k.startswith("gat2.") else k): v
              for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, mcfg


# --------------------------------------------------------------- plotting

# Single oblique view used for every panel (rows of multiple angles were dropped
# in favour of one clear angle). elev, azim -- kept here, not annotated on-plot.
PANEL_VIEW = (20, 60)

HEALTHY_COLOR = "#3b4cc0"          # faint blue, both panels
FIB_TRUTH_COLOR = "#b40426"        # ground-truth fibrosis  (red)
FIB_PRED_COLOR = "#ff7f0e"         # predicted fibrosis      (orange) -- distinct
                                   # from GT so a reader instantly sees a mismatch

# Presentation font sizes (bumped up for slides / paper figures).
AXIS_LABEL_FS = 18
TICK_FS = 13
TITLE_FS = 23
GROUP_FS = 23
ROW_FS = 23
LEGEND_FS = 23


def _add_fibrosis_legend(fig, fontsize=LEGEND_FS):
    """Bottom-centre legend (acts as the figure key): red = GT fibrosis,
    orange = predicted fibrosis, blue = healthy in both GT and prediction."""
    from matplotlib.lines import Line2D

    def dot(color, lab):
        return Line2D([0], [0], marker="o", linestyle="none", markersize=13,
                      markerfacecolor=color, markeredgecolor="none", label=lab)

    handles = [dot(FIB_TRUTH_COLOR, "Ground truth fibrosis"),
               dot(FIB_PRED_COLOR, "Predicted fibrosis"),
               dot(HEALTHY_COLOR, "Ground truth & predicted healthy")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=fontsize, bbox_to_anchor=(0.5, 0.0))


def _render_surface_panel(ax, tm_xyz, vals, kind, elec_xyz=None,
                          scored_mask=None, elec_scored_mask=None, view=PANEL_VIEW):
    """Draw one 3D panel of TM nodes coloured by ``vals`` (0=healthy, 1=fibrotic).

    kind          : "truth" or "pred" -- only changes the fibrotic colour so GT
                    and prediction panels are visually distinguishable.
    scored_mask   : if given, TM nodes with False are guard-band-excluded and are
                    NOT drawn at all (hard removal, not greyed out).
    elec_scored_mask : likewise for electrodes; excluded electrodes are not drawn.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    tm_xyz = np.asarray(tm_xyz, dtype=float)
    vals = np.asarray(vals)
    if scored_mask is None:
        scored_mask = np.ones(tm_xyz.shape[0], dtype=bool)
    scored_mask = np.asarray(scored_mask, dtype=bool)
    fib_color = FIB_TRUTH_COLOR if kind == "truth" else FIB_PRED_COLOR

    heal = (vals == 0) & scored_mask
    fib = (vals == 1) & scored_mask
    ax.scatter(tm_xyz[heal, 0], tm_xyz[heal, 1], tm_xyz[heal, 2],
               c=HEALTHY_COLOR, s=6, alpha=0.20, label="healthy")
    ax.scatter(tm_xyz[fib, 0], tm_xyz[fib, 1], tm_xyz[fib, 2],
               c=fib_color, s=30, alpha=0.95, label="fibrotic")
    if elec_xyz is not None:
        elec_xyz = np.asarray(elec_xyz, dtype=float)
        if elec_scored_mask is None:
            elec_scored_mask = np.ones(elec_xyz.shape[0], dtype=bool)
        elec_scored_mask = np.asarray(elec_scored_mask, dtype=bool)
        ek = elec_xyz[elec_scored_mask]
        ax.scatter(ek[:, 0], ek[:, 1], ek[:, 2], c="k", s=6, marker="^", alpha=0.35)
    ax.view_init(elev=view[0], azim=view[1])
    # Fixed x ticks so the denser (more-curved) panels don't overlap-label; both
    # geometries' x ranges sit within +/-13, so these five ticks read cleanly.
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlabel("x", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("y", fontsize=AXIS_LABEL_FS)
    ax.set_zlabel("z", fontsize=AXIS_LABEL_FS)


def plot_surface_overlay(tm_xyz, elec_xyz, preds, labels, out_path,
                         thr_tag="", scored_mask=None, elec_scored_mask=None):
    """Single-view 3D scatter of TM nodes: ground-truth fibrosis (left) vs
    prediction (right), electrodes overlaid.

    One viewing angle, two columns titled "Ground truth" / "Prediction". Healthy
    nodes are faint blue and small; fibrotic nodes large, drawn red for GT and
    orange for prediction so a mismatch is obvious. ``preds`` should already be
    the binary class at the chosen operating threshold (not argmax@0.5).

    Guard band: ``scored_mask`` / ``elec_scored_mask`` False entries are removed
    from BOTH panels entirely (nodes and electrodes inside the rim are not drawn).
    """
    panels = [("Ground truth", np.asarray(labels), "truth"),
              ("Prediction", np.asarray(preds), "pred")]

    fig = plt.figure(figsize=(9, 5.2))
    for c, (title, vals, kind) in enumerate(panels):
        ax = fig.add_subplot(1, 2, c + 1, projection="3d")
        _render_surface_panel(ax, tm_xyz, vals, kind, elec_xyz=elec_xyz,
                              scored_mask=scored_mask,
                              elec_scored_mask=elec_scored_mask)
        ax.set_title(title, fontsize=TITLE_FS)
    if thr_tag:
        fig.suptitle(thr_tag.strip(" @"), y=0.99, fontsize=TICK_FS)
    _add_fibrosis_legend(fig)            # colour key at the bottom
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- thresholding

def best_f1_threshold(y_true, y_prob):
    """Threshold on the fibrotic probability that maximises F1 on the
    precision-recall curve. Returns (threshold, precision, recall, f1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr) == len(prec) - 1; align by dropping
    # the trailing (recall=0) point that has no threshold.
    prec, rec = prec[:-1], rec[:-1]
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    if f1.size == 0:
        return 0.5, 0.0, 0.0, 0.0
    i = int(np.argmax(f1))
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])


def param_edge_distance(coords_tp, ref_lo=None, ref_hi=None):
    """Per-point normalised distance-to-rim in (theta,phi): 0 at the parameter
    rectangle boundary, ~1 at its centre. ``ref_lo``/``ref_hi`` give the
    reference bounding box (each a length-2 array of [theta,phi] min/max); when
    omitted the box is taken from ``coords_tp`` itself. Passing a shared box (the
    TM bbox) lets electrodes and TM nodes use the SAME rim reference."""
    coords_tp = np.asarray(coords_tp, dtype=float)
    lo = coords_tp.min(0) if ref_lo is None else np.asarray(ref_lo, dtype=float)
    hi = coords_tp.max(0) if ref_hi is None else np.asarray(ref_hi, dtype=float)

    def nd(v, lo_i, hi_i):
        return np.minimum(v - lo_i, hi_i - v) / max(0.5 * (hi_i - lo_i), 1e-9)

    return np.minimum(nd(coords_tp[:, 0], lo[0], hi[0]),
                      nd(coords_tp[:, 1], lo[1], hi[1]))


def tm_edge_distance(meta):
    """Per-TM-node distance-to-rim (see param_edge_distance). Order matches
    g.y[tm_mask]."""
    tp = np.asarray(meta["tm_coords"], dtype=float) / float(meta.get("coord_scale", 1.0))
    return param_edge_distance(tp)


def elec_edge_distance(meta, elec_coords):
    """Per-electrode distance-to-rim, using the TM parameter bbox as the shared
    reference so electrodes and TM nodes are guard-banded against the same rim."""
    cs = float(meta.get("coord_scale", 1.0))
    tm = np.asarray(meta["tm_coords"], dtype=float) / cs
    ec = np.asarray(elec_coords, dtype=float) / cs
    return param_edge_distance(ec, ref_lo=tm.min(0), ref_hi=tm.max(0))


def gaussian_curvature_ellipsoid(xyz, rx, ry, rz):
    """Gaussian curvature K at surface points of an ellipsoid with semi-axes
    (rx,ry,rz):  K = 1 / (rx^2 ry^2 rz^2 (x^2/rx^4 + y^2/ry^4 + z^2/rz^4)^2).
    Larger axes -> smaller K (flatter)."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    s = x**2 / rx**4 + y**2 / ry**4 + z**2 / rz**4
    return 1.0 / (rx**2 * ry**2 * rz**2 * np.maximum(s, 1e-30) ** 2)


def max_gaussian_curvature_over_patch(meta, labels):
    """Max Gaussian curvature over the fibrotic patch (labels==1) of a surface
    graph -- a single scalar quantifying how curved the geometry is where it
    matters for detection."""
    xyz = np.asarray(meta["tm_xyz"], dtype=float)
    patch = np.asarray(labels) == 1
    if not patch.any():
        patch = np.ones(xyz.shape[0], dtype=bool)
    K = gaussian_curvature_ellipsoid(xyz[patch], float(meta["rx"]),
                                     float(meta["ry"]), float(meta["rz"]))
    return float(np.max(K))


def report_at_threshold(y_true, y_prob, thr, tag):
    """Print the classification report + confusion at a given probability
    threshold (fibrotic if prob >= thr)."""
    y_pred = (y_prob >= thr).astype(int)
    print(f"\n--- TM nodes @ threshold={thr:.4f} ({tag}) ---")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["healthy", "fibrotic"],
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"confusion [[TN FP],[FN TP]] = {cm.tolist()}")
    return y_pred


# --------------------------------------------------------------- paper figure

def _eval_graph(model, graph_path, device):
    """Run a loaded SurfaceCNN_GAT on one graph; return everything a panel needs."""
    g = torch.load(graph_path, map_location=device, weights_only=False)
    with torch.no_grad():
        logits, _ = model(
            g.x.to(device), g.edge_index.to(device),
            getattr(g, "edge_attr", None),
            tm_mask=g.tm_mask.to(device),
            elec_coords=g["elec_coords"],
            tm_coords=g["meta"]["tm_coords"],
        )
    tm = g.tm_mask.bool()
    meta = g["meta"]
    return {
        "probs": F.softmax(logits[tm], dim=1)[:, 1].cpu().numpy(),
        "labels": g.y[tm].cpu().numpy(),
        "tm_xyz": np.asarray(meta["tm_xyz"], dtype=float),
        "elec_xyz": np.asarray(meta["elec_xyz"], dtype=float),
        "edge_dist": tm_edge_distance(meta),
        "elec_dist": elec_edge_distance(meta, g["elec_coords"]),
        "max_K": max_gaussian_curvature_over_patch(meta, g.y[tm].cpu().numpy()),
    }


def _panel_pred(res, guard_band, threshold):
    """Binary prediction for a panel: user threshold if given, else max-F1 on the
    scored (interior) nodes of this graph."""
    keep = (res["labels"] != -100) & (res["edge_dist"] >= guard_band)
    if threshold is not None:
        thr = threshold
    elif np.unique(res["labels"][keep]).size >= 2:
        thr, *_ = best_f1_threshold(res["labels"][keep], res["probs"][keep])
    else:
        thr = 0.5
    return (res["probs"] >= thr).astype(int), thr


def build_paper_grid(zeroshot_ckpt, finetuned_ckpt, graph_less, graph_more,
                     out_path, device, guard_band=0.0, threshold=None):
    """One 2x4 figure: rows = [zero-shot, fine-tuned] models, columns =
    [GT, pred] for the less-curved geometry then [GT, pred] for the more-curved.
    Ground truth is model-independent, so the GT columns repeat across rows."""
    models = [("Zero-shot", load_surface_model(zeroshot_ckpt, device)[0]),
              ("Fine-tuned", load_surface_model(finetuned_ckpt, device)[0])]
    geoms = [("Less curved", graph_less), ("More curved", graph_more)]

    fig = plt.figure(figsize=(18, 9))
    max_K = {}
    for r, (row_label, model) in enumerate(models):
        for gi, (geom_label, gpath) in enumerate(geoms):
            res = _eval_graph(model, gpath, device)
            max_K[geom_label] = res["max_K"]
            scored = res["edge_dist"] >= guard_band
            elec_scored = res["elec_dist"] >= guard_band
            preds, _ = _panel_pred(res, guard_band, threshold)
            # columns: GT then pred for this geometry -> base col = gi*2
            for k, (kind, vals, title) in enumerate(
                    [("truth", res["labels"], "Ground truth"),
                     ("pred", preds, "Prediction")]):
                col = gi * 2 + k
                ax = fig.add_subplot(2, 4, r * 4 + col + 1, projection="3d")
                _render_surface_panel(ax, res["tm_xyz"], vals, kind,
                                      elec_xyz=res["elec_xyz"], scored_mask=scored,
                                      elec_scored_mask=elec_scored)
                if r == 0:
                    ax.set_title(title, fontsize=TITLE_FS)
                if col == 0:                      # row label on the left edge
                    ax.text2D(-0.25, 0.5, row_label, transform=ax.transAxes,
                              rotation=90, va="center", ha="center",
                              fontsize=ROW_FS, fontweight="bold")

    # Column-group headers spanning each geometry's two columns (matplotlib has
    # no native column spanner, so place figure-level text at the pair centres).
    # Curvature only -- the reader infers "less/more curved" from the value.
    for gi, (geom_label, _) in enumerate(geoms):
        x = 0.30 + 0.42 * gi          # ~centre of columns {0,1} and {2,3}
        fig.text(x, 0.975, f"Max Gaussian curvature = {max_K[geom_label]:.2e}",
                 ha="center", va="top", fontsize=GROUP_FS, fontweight="bold")

    _add_fibrosis_legend(fig)            # colour key at the bottom
    # top=0.925 leaves a moderate gap between the curvature headers (y=0.975) and
    # the per-column "Ground truth"/"Prediction" titles atop each subplot.
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.925))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote paper grid to {out_path} "
          f"(max K: {', '.join(f'{k}={v:.3g}' for k, v in max_K.items())}).")


# --------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gat-ckpt", required=True,
                   help="best_model.pt from train_gnn_with_pretrained_cnn.py. "
                        "In --paper-grid mode this is the ZERO-SHOT checkpoint.")
    p.add_argument("--graphs-dir",
                   help="Directory with graph_index.txt + .pt graphs from "
                        "surface_to_graph.py. Required unless --paper-grid is set.")
    p.add_argument("--out-dir", default="surface_eval")
    p.add_argument("--n-overlays", type=int, default=5)
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed probability threshold for the fibrotic class. "
                        "Default reports argmax@0.5 AND the max-F1 threshold "
                        "found on this data; pass a value to also apply it.")
    p.add_argument("--guard-band", type=float, default=0.0,
                   help="Exclude TM nodes within this normalised distance-to-rim "
                        "(0=keep all; e.g. 0.2 drops the outer rim) from ALL "
                        "metrics + threshold selection. Quantifies the "
                        "interior-only ceiling, isolating the boundary artifact.")
    # --- 2x4 paper figure mode (zero-shot vs fine-tuned x less/more curved) ---
    p.add_argument("--paper-grid", default=None,
                   help="Output PNG path. When set, build the 2x4 comparison "
                        "figure instead of the normal single-dir evaluation.")
    p.add_argument("--finetuned-ckpt", help="Fine-tuned checkpoint (row 2).")
    p.add_argument("--graph-less", help="Less-curved surface graph .pt.")
    p.add_argument("--graph-more", help="More-curved surface graph .pt.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.paper_grid:
        missing = [n for n in ("finetuned_ckpt", "graph_less", "graph_more")
                   if getattr(args, n) is None]
        if missing:
            p.error("--paper-grid needs --" + ", --".join(m.replace("_", "-")
                    for m in missing))
        build_paper_grid(args.gat_ckpt, args.finetuned_ckpt, args.graph_less,
                         args.graph_more, args.paper_grid, device,
                         guard_band=args.guard_band, threshold=args.threshold)
        return

    if not args.graphs_dir:
        p.error("--graphs-dir is required (unless --paper-grid is set).")

    model, mcfg = load_surface_model(args.gat_ckpt, device)
    print(f"Loaded model: tm_feature_mode={mcfg.get('tm_feature_mode')}, "
          f"embed_dim={mcfg.get('embed_dim')}")

    index_file = Path(args.graphs_dir) / "graph_index.txt"
    if not index_file.exists():
        raise FileNotFoundError(f"Missing {index_file}; run surface_to_graph.py first.")
    graph_paths = [Path(l.strip()) for l in index_file.read_text().splitlines() if l.strip()]
    print(f"{len(graph_paths)} surface graph(s).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: run the model on every graph, stash per-graph results so overlays
    # can be drawn AFTER we know the global operating threshold.
    records = []
    all_preds, all_labels, all_probs = [], [], []
    n_dropped = 0
    for gp in graph_paths:
        g = torch.load(gp, map_location=device, weights_only=False)
        with torch.no_grad():
            logits, _ = model(
                g.x.to(device), g.edge_index.to(device),
                getattr(g, "edge_attr", None),
                tm_mask=g.tm_mask.to(device),
                elec_coords=g["elec_coords"],
                tm_coords=g["meta"]["tm_coords"],
            )
        tm = g.tm_mask.bool()
        tm_logits = logits[tm]
        preds = tm_logits.argmax(dim=1).cpu().numpy()
        probs = F.softmax(tm_logits, dim=1)[:, 1].cpu().numpy()
        labels = g.y[tm].cpu().numpy()
        meta = g["meta"]

        # Guard-band: drop the outer rim from scoring (boundary-artifact ceiling).
        edge_dist = tm_edge_distance(meta)
        keep = (labels != -100) & (edge_dist >= args.guard_band)
        n_dropped += int((labels != -100).sum() - keep.sum())

        all_preds.append(preds[keep])
        all_labels.append(labels[keep])
        all_probs.append(probs[keep])

        records.append({
            "gp": gp,
            "tm_xyz": np.asarray(meta["tm_xyz"], dtype=float),
            "elec_xyz": np.asarray(meta["elec_xyz"], dtype=float),
            "probs": probs,
            "labels": labels,
            "edge_dist": edge_dist,
            "elec_dist": elec_edge_distance(meta, g["elec_coords"]),
        })

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    print("\n=== Surface transfer performance (TM nodes) ===")
    if args.guard_band > 0:
        print(f"guard-band={args.guard_band:g}: scored {len(y_true)} interior TM "
              f"nodes, excluded {n_dropped} rim nodes.")
    print("--- argmax @ 0.5 (default operating point) ---")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["healthy", "fibrotic"],
                                zero_division=0))
    have_two = np.unique(y_true).size >= 2
    if have_two:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f} | "
              f"AP: {average_precision_score(y_true, y_prob):.4f} "
              f"(prevalence {y_true.mean():.4f})")

    # Ranking is geometry-robust but the 0.5 cut is calibrated for the flat
    # domain; recover a usable operating point by maximising F1 on this data.
    cm_pred = y_pred  # what the saved confusion matrix plots
    cm_thr_tag = "argmax 0.5"
    op_thr = 0.5       # operating threshold used for the overlays
    if have_two:
        thr_f1, p_f1, r_f1, f1 = best_f1_threshold(y_true, y_prob)
        print(f"\nMax-F1 threshold = {thr_f1:.4f}: precision {p_f1:.3f}, "
              f"recall {r_f1:.3f}, F1 {f1:.3f}")
        cm_pred = report_at_threshold(y_true, y_prob, thr_f1, "max-F1")
        cm_thr_tag = f"max-F1 {thr_f1:.3f}"
        op_thr = thr_f1
        if args.threshold is not None:
            cm_pred = report_at_threshold(y_true, y_prob, args.threshold, "user")
            cm_thr_tag = f"user {args.threshold:.3f}"
            op_thr = args.threshold

    cm = confusion_matrix(y_true, cm_pred, labels=[0, 1])
    fig = plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["healthy", "fibrotic"]); plt.yticks([0, 1], ["healthy", "fibrotic"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Surface TM confusion ({cm_thr_tag})")
    plt.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # Pass 2: overlays at the operating threshold (so the prediction panel
    # reflects the reported operating point, not the over-predicting argmax@0.5).
    n_overlay_done = 0
    for rec in records[:args.n_overlays]:
        try:
            preds_bin = (rec["probs"] >= op_thr).astype(int)
            scored = rec["edge_dist"] >= args.guard_band
            elec_scored = rec["elec_dist"] >= args.guard_band
            plot_surface_overlay(
                rec["tm_xyz"], rec["elec_xyz"], preds_bin, rec["labels"],
                out_dir / f"overlay_{rec['gp'].stem}.png",
                thr_tag=f" @ {cm_thr_tag}", scored_mask=scored,
                elec_scored_mask=elec_scored)
            n_overlay_done += 1
        except Exception as e:
            print(f"Overlay failed for {rec['gp'].name}: {e}")

    np.savez(out_dir / "surface_predictions.npz",
             preds=y_pred, labels=y_true, probs=y_prob)
    print(f"\nWrote metrics, {n_overlay_done} overlay(s), and predictions to {out_dir}.")


if __name__ == "__main__":
    main()
