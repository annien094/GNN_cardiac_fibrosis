"""
extrapolate_cnn_to_tm.py

Take a trained per-electrode CNN, classify each electrode (fibrotic/healthy) from its
phie signal, then extrapolate those labels onto the TM-node grid: every TM node inherits
the label of its NEAREST electrode (Voronoi / nearest-neighbour assignment). The electrode
spacing d is derived from `elecpos`; on a regular grid the nearest-electrode rule is exactly
the "TM node inside the d/2 x d/2 square centred on the electrode" rule, and TM nodes that
happen to fall outside every square (only possible with a cropped/sparse electrode set) still
take their nearest electrode's label. The d/2 square test is recorded as `inside_square` for
diagnostics only -- it never changes a label.

TM-node ordering and coordinate space exactly mirror preprocess_graphs.py, so the output
`tm_labels` vector aligns 1:1 with the tm_mask nodes (x[num_elec:]) of the corresponding graph.

Usage:
    python extrapolate_cnn_to_tm.py \
        --ckpt $EPHEMERAL/cnn_electrode_only/cnn_elec_higher_k_thr2_n6000_seed42/best_model.pt \
        --data-glob "AP_simulations_higher_k/*.npz" \
        --out-dir $EPHEMERAL/baseline_cnn_extrapolation/higher_k
"""

import argparse
import glob
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Reuse model + preprocessing so we never drift from training conventions.
from train_cnn_electrode import (
    ElectrodeCNN,
    standardize_phie,
    pad_or_trim,
    labels_from_fiblocs,   # electrode-level GT
)
# Per-TM-node GT, identical to training (gt_label_base reshaped row-major).
from preprocess_graphs import build_gt_label_base

H0, W0 = 100, 100          # domain grid, matches preprocess_graphs.py
ELEC_SCALE = 10.0          # elecpos is divided by 10 in load_sim_file


# ----------------------------- model -----------------------------

class LegacyElectrodeCNN(ElectrodeCNN):
    """ElectrodeCNN as it was when the older 16-d embedding checkpoints (e.g. the
    cnn_electrode_only best_model.pt, and pretrained_elec_cnn_gat_* runs) were trained.

    Mirrors eval_surface_gnn.py::LegacyElectrodeCNN. The conv `features` stack is
    unchanged, but the head was a single Linear (`embed`, no LayerNorm) producing an
    explicit ``embed_dim`` after global temporal pooling, vs today's ``head`` =
    LazyLinear(conv_ch) + LayerNorm. Matches state_dict keys ``embed.1.*`` and
    ``classifier`` = Linear(embed_dim, 2).
    """

    def __init__(self, embed_dim=16, n_classes=2, **kw):
        super().__init__(n_classes=n_classes, **kw)   # builds features/head/classifier
        del self.head                                 # drop the modern head
        # Old embed Linear takes conv_ch-sized input, so the old model pooled over
        # time first. Append a param-free global temporal pool so features.* keys match.
        self.features = nn.Sequential(*self.features, nn.AdaptiveAvgPool1d(1))
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(embed_dim),                 # -> embed.1.* (conv_ch -> embed_dim)
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


def _accepted_kwargs(*classes):
    """Union of explicit (non-var) __init__ params across classes, minus `self`."""
    accepted = set()
    for cls in classes:
        for name, p in inspect.signature(cls.__init__).parameters.items():
            if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                accepted.add(name)
    return accepted


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))  # best_model.pt stores this
    if not cfg:
        # final_model.pt has no model_config; fall back to the run's config.json
        # (train_cnn_electrode.py writes the full arg set there), so we rebuild with
        # the EXACT training hyperparams rather than ElectrodeCNN defaults.
        cfg_path = Path(ckpt_path).parent / "config.json"
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text())
            cfg = {k: raw[k] for k in ("conv_ch", "kernel_size_1", "kernel_size_2")
                   if k in raw}
            if "T_max" in raw:
                cfg["T"] = raw["T_max"]
            print(f"load_model: no model_config in checkpoint; using {cfg_path}")
    state = ckpt["model_state_dict"]
    T = cfg.get("T", 1500)

    # Detect architecture from the checkpoint: legacy 16-d embedding has `embed.*`
    # keys; the modern head-based model has `head.*`.
    is_legacy = any(k.startswith("embed.") for k in state)
    if is_legacy:
        Model = LegacyElectrodeCNN
        accepted = _accepted_kwargs(LegacyElectrodeCNN, ElectrodeCNN)  # incl. embed_dim
    else:
        Model = ElectrodeCNN
        accepted = _accepted_kwargs(ElectrodeCNN)

    dropped = sorted(set(cfg) - accepted)
    if dropped:
        print(f"load_model: ignoring unused model_config keys: {dropped}")
    kw = {k: v for k, v in cfg.items() if k in accepted}
    print(f"load_model: building {Model.__name__} with {kw}")

    model = Model(**kw).to(device)
    # Materialize LazyLinear layers before loading weights.
    with torch.no_grad():
        model(torch.zeros(1, T, device=device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, T


@torch.no_grad()
def predict_electrodes(model, phie, device, T_max, scaling="std"):
    """phie: (N, T) raw -> (preds (N,) int, prob_fibrotic (N,) float)."""
    phie = np.asarray(phie, dtype=np.float32)
    if scaling == "std":
        phie = standardize_phie(phie)
    elif scaling == "min-max":
        mn, mx = phie.min(), phie.max()
        phie = (phie - mn) / max(float(mx - mn), 1e-6)
    phie = pad_or_trim(phie, T_max)
    x = torch.from_numpy(phie).float().to(device)        # (N, T)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


# ----------------------------- geometry (mirrors preprocess_graphs.py) -----------------------------

def build_tm_coords(target_h, target_w):
    """Full-domain target_h x target_w TM grid in the corrected (row=axis0, col=axis1) frame.

    Mirrors preprocess_graphs.py's TM grid: a regular grid over the FULL domain (0..W0-1,
    not the electrode bbox) at the target resolution, so the baseline is scored on the SAME
    TM nodes as the GNN graphs at that resolution (set --target-h/--target-w to match the
    graph dir you compare against). Coords are in elec_coords units (elecpos/ELEC_SCALE), so
    the full domain spans [0, (H0-1)/ELEC_SCALE]; node i = (row=i//target_w, col=i%target_w).

    NB: corrected (row,col) frame -- NOT preprocess's historical meshgrid(xg,yg) transpose;
    see memory project_electrode_cnn_tm_transpose / project_overlay_coord_conventions.

    Returns (tm_coords (target_h*target_w, 2), (row_min, row_max, col_min, col_max)).
    """
    row_max = (H0 - 1) / ELEC_SCALE
    col_max = (W0 - 1) / ELEC_SCALE
    row_coords = np.linspace(0.0, row_max, target_h)
    col_coords = np.linspace(0.0, col_max, target_w)
    # indexing='ij': RR[a,b]=row_coords[a], CC[a,b]=col_coords[b]; ravel -> node i=a*target_w+b.
    RR, CC = np.meshgrid(row_coords, col_coords, indexing="ij")
    tm_coords = np.column_stack([RR.ravel(), CC.ravel()])    # (axis0=row, axis1=col)
    return tm_coords, (0.0, row_max, 0.0, col_max)


def electrode_spacing(elec_coords):
    """Spacing d = median nearest-neighbour distance between electrodes (robust to grid edges)."""
    diff = elec_coords[:, None, :] - elec_coords[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn_dist = np.sqrt(d2.min(axis=1))
    return float(np.median(nn_dist))


def extrapolate_to_tm(tm_coords, elec_coords, elec_values, d):
    """Each TM node takes its NEAREST electrode's value.

    elec_values may be hard labels (int) or probabilities (float).
    `inside_square` flags whether the TM node lies within the d/2 x d/2 square of its
    nearest electrode -- diagnostic only; it does not alter the assigned value.

    Returns (tm_values (Ntm,), nn_idx (Ntm,), inside_square (Ntm,)).
    """
    diff = tm_coords[:, None, :] - elec_coords[None, :, :]   # (Ntm, Nelec, 2)
    d2 = np.sum(diff ** 2, axis=-1)
    nn_idx = np.argmin(d2, axis=1)                           # nearest electrode

    rows = np.arange(tm_coords.shape[0])
    sel = diff[rows, nn_idx]                                 # signed offset to chosen electrode
    inside = np.all(np.abs(sel) <= d / 2.0 + 1e-9, axis=1)

    tm_values = elec_values[nn_idx]
    return tm_values, nn_idx, inside


# ----------------------------- plotting -----------------------------

def _full_domain_pred_map(elec_coords, elec_values):
    """Nearest-electrode (Voronoi) prediction over the FULL H0xW0 cell grid, so the
    baseline plots in the same full-domain frame as the GT and as
    plot_gnn_from_ckpt -- electrodes then sit INSET (margin = half the electrode
    spacing) instead of stretched to the panel edge. elec_values: per-electrode
    hard labels (or probabilities)."""
    elec_cells = np.asarray(elec_coords, dtype=float) * ELEC_SCALE   # -> cell units
    rr, cc = np.meshgrid(np.arange(H0), np.arange(W0), indexing="ij")
    grid = np.column_stack([rr.ravel(), cc.ravel()]).astype(float)
    d2 = np.sum((grid[:, None, :] - elec_cells[None, :, :]) ** 2, axis=-1)
    nn = d2.argmin(axis=1)
    return np.asarray(elec_values)[nn].reshape(H0, W0)


def render_overlay_row(results, out_path, fontsize=16, titles=None):
    """One row of panels (one per sim): nearest-electrode prediction (filled) + GT contour
    + electrodes + stimuli. Matches plot_gnn_from_ckpt.render_overlay_row formatting:
    ticks off, a single shared legend, paper-readable fonts.

    Everything is drawn in the full-domain (H0xW0 cell) frame: the prediction is the
    nearest-electrode Voronoi map over the whole domain, the GT is the per-cell GT, and
    electrodes/stimuli use full-domain cell coordinates -- so electrodes sit inset with a
    margin, exactly like plot_gnn_from_ckpt (and the two methods are spatially comparable).

    `titles` (optional): per-panel column titles, e.g. the parameter case for a
    cross-case comparison figure. Omit for the default no-title look.
    """
    ncols = len(results)
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.4 if titles else 3.2),
                             squeeze=False)
    axes = axes[0]
    side = H0
    denom = max(1, side - 1)
    for i, (ax, res) in enumerate(zip(axes, results)):
        pr = _full_domain_pred_map(res["elec_coords"], res["elec_preds"])
        gt = res["tm_gt_map"]
        ax.imshow(pr, origin="lower", cmap="Blues", vmin=0, vmax=1)
        if gt.min() < 0.5 < gt.max():
            ax.contour(gt, levels=[0.5], colors="red", linewidths=2.0, origin="lower")
        ec_cells = np.asarray(res["elec_coords"], dtype=float) * ELEC_SCALE  # cell units
        ax.scatter(ec_cells[:, 1] / denom * (side - 1),
                   ec_cells[:, 0] / denom * (side - 1),
                   s=12, c="white", edgecolors="black", linewidths=0.4, zorder=4)
        stim = res.get("stim")
        if stim is not None:
            stim = np.asarray(stim, dtype=float).reshape(-1, 2)   # (row, col) cell units
            if stim.size:
                ax.scatter(stim[:, 1] / denom * (side - 1),
                           stim[:, 0] / denom * (side - 1),
                           s=80, marker="X", c="cyan",
                           edgecolors="black", linewidths=0.8, zorder=5)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=fontsize)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    handles = [
        Patch(facecolor=plt.cm.Blues(0.85), edgecolor="none", label="Prediction"),
        Line2D([], [], color="red", lw=2.0, label="Ground truth"),
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor="white", markeredgecolor="black", label="Electrodes"),
        Line2D([], [], marker="X", linestyle="none", markersize=11,
               markerfacecolor="cyan", markeredgecolor="black", label="Stimuli"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=fontsize,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote overlay row ({ncols} panels) to {out_path}")


# ----------------------------- metrics -----------------------------

def compute_and_save_metrics(out_dir, gt, preds, probs, prefix,
                             pos_name="fibrotic", neg_name="healthy"):
    """Classification report + confusion matrix + threshold-free & imbalance-robust metrics.

    Threshold-free (use probabilities): ROC AUC, average precision (AP), PR-AUC (trapezoidal
    area under the precision-recall curve), Brier score (calibration).
    Threshold-based (use hard preds): accuracy, balanced accuracy, Matthews correlation
    coefficient (MCC), Cohen's kappa -- all more informative than raw accuracy under imbalance.

    Writes <prefix>_classification_report.txt, <prefix>_metrics.json,
    <prefix>_confusion_matrix.png and (if both classes present) <prefix>_roc_pr_curves.png.
    Returns the metrics dict.
    """
    gt = np.asarray(gt).astype(int)
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs, dtype=float)

    report = classification_report(
        gt, preds, labels=[0, 1], target_names=[neg_name, pos_name], zero_division=0,
    )
    cm = confusion_matrix(gt, preds, labels=[0, 1])

    both_classes = (gt == 0).any() and (gt == 1).any()
    # Threshold-free curves/areas need both classes present.
    fpr = tpr = prec = rec = None
    if both_classes:
        fpr, tpr, _ = roc_curve(gt, probs)
        prec, rec, _ = precision_recall_curve(gt, probs)

    metrics = {
        "n": int(gt.size),
        "n_pos": int((gt == 1).sum()),
        "n_neg": int((gt == 0).sum()),
        "prevalence": float((gt == 1).mean()) if gt.size else None,
        "confusion_matrix": cm.tolist(),
        # threshold-based
        "accuracy": float((preds == gt).mean()) if gt.size else None,
        "balanced_accuracy": float(balanced_accuracy_score(gt, preds)) if both_classes else None,
        "mcc": float(matthews_corrcoef(gt, preds)) if both_classes else None,
        "cohen_kappa": float(cohen_kappa_score(gt, preds)) if both_classes else None,
        # threshold-free
        "roc_auc": float(roc_auc_score(gt, probs)) if both_classes else None,
        "average_precision": float(average_precision_score(gt, probs)) if both_classes else None,
        "pr_auc": float(auc(rec, prec)) if both_classes else None,
        "brier_score": float(brier_score_loss(gt, probs)) if both_classes else None,
    }
    if not both_classes:
        print(f"[{prefix}] only one GT class present -> AUC/AP/PR-AUC/MCC etc. undefined (null).")

    print(f"\n=== {prefix} metrics (pooled over {metrics['n']} nodes, "
          f"prevalence={metrics['prevalence']:.4f}) ===")
    print(report)
    print(f"ROC AUC={metrics['roc_auc']}  AP={metrics['average_precision']}  "
          f"PR-AUC={metrics['pr_auc']}")
    print(f"balanced_acc={metrics['balanced_accuracy']}  MCC={metrics['mcc']}  "
          f"kappa={metrics['cohen_kappa']}  Brier={metrics['brier_score']}")

    (out_dir / f"{prefix}_classification_report.txt").write_text(report)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix heatmap (no seaborn dependency).
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=[neg_name, pos_name])
    ax.set_yticks([0, 1], labels=[neg_name, pos_name])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{prefix} confusion matrix")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ROC + PR curves (curves already computed above).
    if both_classes:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC"); axes[0].legend(loc="lower right")
        axes[1].plot(rec, prec,
                     label=f"PR-AUC = {metrics['pr_auc']:.3f}, AP = {metrics['average_precision']:.3f}")
        axes[1].axhline(metrics["prevalence"], color="k", ls="--", lw=0.8,
                        label=f"baseline = {metrics['prevalence']:.3f}")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall"); axes[1].legend(loc="upper right")
        fig.suptitle(f"{prefix}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_roc_pr_curves.png", dpi=120)
        plt.close(fig)

    return metrics


# ----------------------------- per-file driver -----------------------------

def process_file(path, model, T_max, device, scaling="std",
                 fib_threshold=2.0, target_h=30, target_w=30):
    d = np.load(path, allow_pickle=True)
    phie = np.asarray(d["phie"], dtype=np.float32)            # (N, T)
    elecpos = np.asarray(d["elecpos"], dtype=np.float64)      # (N, 2), raw grid units

    elec_coords = elecpos / ELEC_SCALE                       # graph coordinate space
    preds, probs = predict_electrodes(model, phie, device, T_max, scaling=scaling)

    tm_coords, bbox = build_tm_coords(target_h, target_w)
    spacing = electrode_spacing(elec_coords)

    tm_labels, nn_idx, inside = extrapolate_to_tm(tm_coords, elec_coords, preds, spacing)
    tm_probs, _, _ = extrapolate_to_tm(tm_coords, elec_coords, probs, spacing)

    # GT: build at full 100x100 then nearest-neighbour downsample to the target resolution
    # (matches preprocess_graphs.py's resize(order=0) > 0.5), so tm_labels_gt aligns with the
    # GNN graph's y at this resolution. Keep the full-res GT (gt_full) for the overlay contour.
    gt_full = build_gt_label_base(d["fiblocs"], H0, W0, path_for_debug=path)   # (100, 100)
    gt_down = resize(gt_full, (target_h, target_w), order=0,
                     preserve_range=True, anti_aliasing=False)
    tm_gt = (gt_down > 0.5).astype(np.int64).reshape(-1)
    elec_gt = labels_from_fiblocs(elecpos, d["fiblocs"], fib_threshold)

    res = {
        "tm_labels": tm_labels,                       # (target_h*target_w,) graph TM nodes
        "tm_probs": tm_probs,                         # soft version
        "tm_labels_gt": tm_gt,                        # per-TM-node GT at target resolution
        "tm_label_map": tm_labels.reshape(target_h, target_w),   # map[row, col]
        "tm_prob_map": tm_probs.reshape(target_h, target_w),
        "tm_gt_map": gt_full.astype(np.int64),        # 100x100 GT, for the overlay contour
        "tm_coords": tm_coords,
        "elec_coords": elec_coords,
        "elec_preds": preds,
        "elec_probs": probs,
        "elec_labels_gt": elec_gt,
        "nn_electrode": nn_idx,
        "spacing_d": spacing,
        "inside_square": inside,
        "bbox": np.asarray(bbox),
    }
    # Stimulus sites (row, col) in cell units, for plotting overlays.
    if "stim_centers" in d.files:
        res["stim"] = np.asarray(d["stim_centers"])
    elif "stim_center" in d.files:
        res["stim"] = np.asarray(d["stim_center"])
    else:
        res["stim"] = None
    res["_fiblocs"] = d["fiblocs"]    # kept locally for plotting; not saved
    return res


def main():
    p = argparse.ArgumentParser(description="Extrapolate per-electrode CNN preds onto TM-node grid.")
    p.add_argument("--ckpt", required=True, help="Path to best_model.pt / final_model.pt")
    p.add_argument("--data-glob", default=None,
                   help="Sim .npz glob. Default: the CNN run's config.json data_glob, so the "
                        "train/val split matches train_cnn_electrode.py exactly.")
    p.add_argument("--split", choices=["val", "all"], default="val",
                   help="'val' (default) reconstructs train_cnn_electrode.py's held-out val "
                        "split and scores ONLY those graphs; 'all' scores every matched graph.")
    p.add_argument("--config", default=None,
                   help="CNN run config.json (default: alongside --ckpt); supplies "
                        "data_glob/seed/val_frac/max_files for the split.")
    p.add_argument("--val-frac", type=float, default=None,
                   help="Override val fraction (else config.json, default 0.2).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override split seed (else config.json, default 42).")
    p.add_argument("--out-dir", default="cnn_tm_extrapolation")
    p.add_argument("--max-files", type=int, default=None,
                   help="Cap matched files BEFORE the split (mirrors train_cnn_electrode "
                        "--max-files; else taken from config.json).")
    p.add_argument("--plot-indices", default=None,
                   help="Comma-separated simulation indices to PLOT (e.g. '1389,11262,11371'); "
                        "matches files whose stem ends in '_<index>'. Overrides --plot-random. "
                        "Metrics are always over ALL processed graphs -- this only picks which "
                        "graphs get an overlay panel.")
    p.add_argument("--plot-random", type=int, default=5,
                   help="If --plot-indices is not given, randomly pick this many graphs from the "
                        "(val) set to plot (0 to skip plotting). The output filename lists the "
                        "chosen indices.")
    p.add_argument("--plot-seed", type=int, default=0,
                   help="Seed for the random plot-graph selection.")
    p.add_argument("--scaling", default="std", choices=["std", "min-max", "none"])
    p.add_argument("--fib-threshold", type=float, default=2.0,
                   help="Nearest-electrode distance threshold for electrode-level GT.")
    p.add_argument("--target-h", type=int, default=30,
                   help="TM-grid rows; set to match the GNN graphs you compare against "
                        "(30 for *_fixed, 60 for the *_60 graphs). Was hard-coded 100 before.")
    p.add_argument("--target-w", type=int, default=30, help="TM-grid columns (see --target-h).")
    p.add_argument("--no-plots", action="store_true", help="Skip overlay plots.")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip pooled classification report / ROC AUC / AP over processed files.")
    p.add_argument("--save-npz", action="store_true",
                   help="Save a per-graph <stem>_tm_pred.npz (TM labels/probs/GT etc.). "
                        "Off by default -- normally only the pooled metrics are kept.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T_max = load_model(args.ckpt, device)

    # Reproduce train_cnn_electrode.py's split EXACTLY: sorted glob -> optional max_files cap ->
    # np.random.default_rng(seed).permutation -> the first n_val = round(val_frac*N) are val.
    cfg_path = Path(args.config) if args.config else Path(args.ckpt).parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if cfg:
        print(f"Loaded split config from {cfg_path}.")
    elif args.split == "val":
        print(f"Warning: no config.json at {cfg_path}; relying on --data-glob/--seed/--val-frac.")

    data_glob = args.data_glob or cfg.get("data_glob")
    if not data_glob:
        raise SystemExit("No --data-glob and none in config.json; cannot locate sims.")
    if args.data_glob and cfg.get("data_glob") and args.data_glob != cfg["data_glob"]:
        print(f"Warning: --data-glob '{args.data_glob}' != config '{cfg['data_glob']}'; the val "
              f"split only matches the CNN if the sorted file set is identical.")

    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {data_glob}")
    max_files = args.max_files if args.max_files is not None else cfg.get("max_files")
    if max_files:
        files = files[: int(max_files)]

    if args.split == "val":
        seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
        val_frac = args.val_frac if args.val_frac is not None else float(cfg.get("val_frac", 0.2))
        perm = np.random.default_rng(seed).permutation(len(files))
        n_val = max(1, int(round(val_frac * len(files))))
        files = [files[i] for i in perm[:n_val]]
        print(f"VAL split: {len(files)} held-out graphs (seed={seed}, val_frac={val_frac}); "
              f"metrics computed over these.")
    else:
        print(f"ALL: {len(files)} graphs; metrics computed over these.")

    # Decide which graphs get an overlay panel (metrics still cover every graph). With
    # --plot-indices use those (in order); otherwise randomly pick --plot-random from the
    # (val) set. The output filename lists the chosen indices either way.
    plot_order = []
    if not args.no_plots:
        if args.plot_indices:
            wanted = [s.strip() for s in args.plot_indices.split(",") if s.strip()]
            matched = {idx: next((f for f in files if Path(f).stem.endswith(f"_{idx}")), None)
                       for idx in wanted}
            plot_order = [Path(f).stem for f in matched.values() if f is not None]
            missing = [idx for idx, f in matched.items() if f is None]
            if missing:
                print(f"Warning: no graph matches plot indices {missing}; skipping those.")
        elif args.plot_random > 0:
            n = min(args.plot_random, len(files))
            pick = np.random.default_rng(args.plot_seed).choice(len(files), size=n, replace=False)
            plot_order = [Path(files[i]).stem for i in sorted(pick)]
        print(f"Will plot {len(plot_order)} of {len(files)} graphs.")
    plot_stems = set(plot_order)
    plot_results = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    if plot_stems:
        plot_dir.mkdir(exist_ok=True)

    # Accumulators for pooled metrics across all processed files.
    pooled = {"tm_gt": [], "tm_pred": [], "tm_prob": [],
              "elec_gt": [], "elec_pred": [], "elec_prob": []}

    for f in files:
        res = process_file(
            f, model, T_max, device, scaling=args.scaling,
            fib_threshold=args.fib_threshold,
            target_h=args.target_h, target_w=args.target_w,
        )
        res.pop("_fiblocs", None)

        if Path(f).stem in plot_stems:
            plot_results[Path(f).stem] = res

        if args.save_npz:
            np.savez_compressed(out_dir / f"{Path(f).stem}_tm_pred.npz", **res)

        if not args.no_metrics:
            pooled["tm_gt"].append(res["tm_labels_gt"])
            pooled["tm_pred"].append(res["tm_labels"])
            pooled["tm_prob"].append(res["tm_probs"])
            pooled["elec_gt"].append(res["elec_labels_gt"])
            pooled["elec_pred"].append(res["elec_preds"])
            pooled["elec_prob"].append(res["elec_probs"])

        print(f"{Path(f).name}: d={res['spacing_d']:.3f}, "
              f"elec fib={int(res['elec_preds'].sum())}/{len(res['elec_preds'])}, "
              f"tm fib={int((res['tm_labels'] == 1).sum())}/{res['tm_labels'].size}")

    # One combined overlay row; filename lists the plotted sim indices so the (possibly
    # random) selection is identifiable.
    if plot_results:
        kept = [s for s in plot_order if s in plot_results]
        ordered = [plot_results[s] for s in kept]
        idxs = [s.split("_")[-1] for s in kept]
        fname = "overlay_row_" + "_".join(idxs) + ".png"
        render_overlay_row(ordered, plot_dir / fname)

    if not args.no_metrics and pooled["tm_gt"]:
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        # Electrode-level: the CNN's own classifier performance.
        compute_and_save_metrics(
            metrics_dir,
            np.concatenate(pooled["elec_gt"]),
            np.concatenate(pooled["elec_pred"]),
            np.concatenate(pooled["elec_prob"]),
            prefix="electrode",
        )
        # TM-node-level: performance after nearest-electrode extrapolation.
        compute_and_save_metrics(
            metrics_dir,
            np.concatenate(pooled["tm_gt"]),
            np.concatenate(pooled["tm_pred"]),
            np.concatenate(pooled["tm_prob"]),
            prefix="tm_node",
        )
        print(f"\nMetrics written to {metrics_dir}")


if __name__ == "__main__":
    main()
