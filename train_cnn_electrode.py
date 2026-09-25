"""
Train a small 1D CNN to classify each electrode as fibrotic vs healthy from its phie signal.

Per-electrode labels are derived from `fiblocs` by mapping each fibrotic coordinate to its
nearest electrode (within `fib_distance_threshold`, in grid units), following the convention
in plot_phie_npz.py. TM nodes are ignored entirely — predictions are made per electrode.

Outputs (under <out_root>/<run_name>/):
  - best_model.pt        : best-by-val-loss checkpoint (model_state_dict + meta)
  - final_model.pt       : last-epoch checkpoint
  - losses.png           : train/val loss curves
  - classification_report.txt
  - confusion_matrix.png
  - val_predictions.npz  : per-electrode preds, labels, file paths (for downstream analysis)
  - config.json          : run config

The trained CNN exposes `forward(x, return_embedding=True)` so it can later be used as a
feature encoder before a GNN.
"""

import argparse
import glob
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset


# ----------------------------- labels & data loading -----------------------------

def normalize_fib_coords(fiblocs_obj):
    """Stack patch coord lists from a `fiblocs` object array into an (M, 2) int array."""
    if isinstance(fiblocs_obj, np.ndarray) and fiblocs_obj.dtype == object:
        parts = [np.asarray(x) for x in fiblocs_obj if np.asarray(x).size > 0]
        if not parts:
            return np.empty((0, 2), dtype=int)
        coords = np.vstack(parts)
    else:
        coords = np.asarray(fiblocs_obj)
    return coords.reshape(-1, 2).astype(int)


def labels_from_fiblocs(elecpos, fiblocs, threshold):
    """Per-electrode binary labels via nearest-electrode mapping (see plot_phie_npz.py).

    For each fibrotic coordinate, find the nearest electrode; if the distance is within
    `threshold`, mark that electrode as fibrotic. Returns an (N,) int64 array.
    """
    N = elecpos.shape[0]
    labels = np.zeros(N, dtype=np.int64)
    fib_coords = normalize_fib_coords(fiblocs)
    if fib_coords.size == 0:
        return labels
    diff = fib_coords[:, None, :] - elecpos[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    nn_idx = np.argmin(d2, axis=1)
    nn_dist = np.sqrt(d2[np.arange(d2.shape[0]), nn_idx])
    keep = nn_dist <= threshold
    labels[np.unique(nn_idx[keep])] = 1
    return labels


def standardize_phie(phie):
    """Per-file z-score on the whole phie tensor (matches preprocess_graphs.py 'std' scaling)."""
    eps = 1e-6
    mean = phie.mean()
    std = phie.std()
    if std < eps:
        std = eps
    return (phie - mean) / std


def pad_or_trim(arr, T):
    L = arr.shape[-1]
    if L == T:
        return arr
    if L > T:
        return arr[..., :T]
    pad = np.zeros((*arr.shape[:-1], T - L), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


class ElectrodeFileDataset(Dataset):
    """One item per .npz file: returns (signals (100, T) float32, labels (100,) int64)."""

    def __init__(self, file_paths, T_max=1500, fib_distance_threshold=1.5, scaling="std"):
        self.file_paths = list(file_paths)
        self.T_max = T_max
        self.fib_distance_threshold = fib_distance_threshold
        self.scaling = scaling

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        d = np.load(path, allow_pickle=True)
        phie = np.asarray(d["phie"], dtype=np.float32)  # (N, T)
        elecpos = np.asarray(d["elecpos"], dtype=np.float64)  # (N, 2), grid units
        fiblocs = d["fiblocs"]

        if phie.ndim != 2:
            raise ValueError(f"{path}: phie must be 2D, got {phie.shape}")
        if elecpos.shape[0] != phie.shape[0]:
            raise ValueError(
                f"{path}: elecpos has {elecpos.shape[0]} rows, phie has {phie.shape[0]}."
            )

        labels = labels_from_fiblocs(elecpos, fiblocs, self.fib_distance_threshold)

        if self.scaling == "std":
            phie = standardize_phie(phie)
        elif self.scaling == "min-max":
            mn, mx = phie.min(), phie.max()
            phie = (phie - mn) / max(float(mx - mn), 1e-6)
        elif self.scaling != "none":
            raise ValueError(f"Unknown scaling: {self.scaling}")

        phie = pad_or_trim(phie, self.T_max)

        return (
            torch.from_numpy(phie).float(),                # (N, T)
            torch.from_numpy(labels).long(),               # (N,)
            str(path),
        )


def collate_files(batch):
    """Flatten file-level items into electrode-level mini-batch."""
    sigs = torch.cat([b[0] for b in batch], dim=0)        # (sum_N, T)
    labs = torch.cat([b[1] for b in batch], dim=0)        # (sum_N,)
    file_ids = []
    for i, b in enumerate(batch):
        file_ids.extend([i] * b[0].shape[0])
    return sigs, labs, file_ids, [b[2] for b in batch]


# ----------------------------- per-file overlay plotting -----------------------------

def _plot_file_overlay(file_path, preds, labels, out_path, title_suffix="",
                       pos_name="fibrotic", neg_name="healthy"):
    """Three-panel overlay (GT, Pred, Confusion) for one simulation file."""
    d = np.load(file_path, allow_pickle=True)
    elecpos = np.asarray(d["elecpos"], dtype=float)
    fib_coords = normalize_fib_coords(d["fiblocs"])

    tp = (preds == 1) & (labels == 1)
    tn = (preds == 0) & (labels == 0)
    fp = (preds == 1) & (labels == 0)
    fn = (preds == 0) & (labels == 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _bg(ax):
        if fib_coords.size > 0:
            ax.scatter(fib_coords[:, 1], fib_coords[:, 0],
                       s=12, c="#d62728", alpha=0.18, label=f"{pos_name} patch")

    def _scatter(ax, mask, **kw):
        if mask.any():
            ax.scatter(elecpos[mask, 1], elecpos[mask, 0], **kw)

    ax = axes[0]
    _bg(ax)
    _scatter(ax, labels == 0, s=55, c="#bbbbbb", edgecolors="k", linewidths=0.5, label=f"{neg_name} (GT)")
    _scatter(ax, labels == 1, s=70, c="#d62728", edgecolors="k", linewidths=0.6, label=f"{pos_name} (GT)")
    ax.set_title("Ground truth")

    ax = axes[1]
    _bg(ax)
    _scatter(ax, preds == 0, s=55, c="#bbbbbb", edgecolors="k", linewidths=0.5, label=f"{neg_name} (pred)")
    _scatter(ax, preds == 1, s=70, c="#1f77b4", edgecolors="k", linewidths=0.6, label=f"{pos_name} (pred)")
    ax.set_title("Prediction")

    ax = axes[2]
    _bg(ax)
    _scatter(ax, tn, s=45, c="#dddddd", edgecolors="k", linewidths=0.4, label=f"TN ({int(tn.sum())})")
    _scatter(ax, tp, s=80, c="#2ca02c", edgecolors="k", linewidths=0.6, label=f"TP ({int(tp.sum())})")
    _scatter(ax, fp, s=80, c="#ff7f0e", edgecolors="k", linewidths=0.6, label=f"FP ({int(fp.sum())})")
    _scatter(ax, fn, s=80, c="#9467bd", edgecolors="k", linewidths=0.6, label=f"FN ({int(fn.sum())})")
    ax.set_title("Confusion (TP/TN/FP/FN)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(f"{Path(file_path).name}{title_suffix}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_random_val_overlays(out_dir, val_preds, val_labels, val_files, n=5, seed=0,
                              pos_name="fibrotic", neg_name="healthy"):
    """Pick `n` random validation files and write a 3-panel overlay PNG for each."""
    files_arr = np.asarray(val_files)
    unique_files = np.unique(files_arr)
    if len(unique_files) == 0:
        return []
    rng = np.random.default_rng(seed)
    n_pick = min(n, len(unique_files))
    chosen = rng.choice(unique_files, size=n_pick, replace=False)

    overlay_dir = Path(out_dir) / "val_overlays"
    overlay_dir.mkdir(exist_ok=True)
    written = []
    for f in chosen:
        mask = files_arr == f
        p_f = val_preds[mask]
        y_f = val_labels[mask]
        suffix = f"  |  GT {pos_name}={int((y_f == 1).sum())}, pred {pos_name}={int((p_f == 1).sum())}"
        out_path = overlay_dir / f"overlay_{Path(f).stem}.png"
        try:
            _plot_file_overlay(f, p_f, y_f, out_path, title_suffix=suffix,
                               pos_name=pos_name, neg_name=neg_name)
            written.append(out_path)
        except Exception as e:
            print(f"Overlay failed on {f}: {e}")
    return written


# ----------------------------- label-name presets -----------------------------

# Maps the patch type to (positive_class_name, negative_class_name) used in
# reports, confusion matrices, overlays, and the default run_name slug. The
# binary label semantics are the same across modes — only the strings change.
LABEL_NAME_PRESETS = {
    "fibrosis":  ("fibrotic", "healthy"),   # lower D
    "higher_k":  ("higher_k", "normal"),    # heightened excitability via higher k
    "lower_a":     ("lower_a",    "normal"),    # heightened excitability via negative a
}


# ----------------------------- model -----------------------------

class ElectrodeCNN(nn.Module):
    """Small 1D CNN for per-electrode fibrosis classification.

    Forward accepts (B, T) or (B, 1, T). With `return_embedding=True`, also returns the
    pre-classifier embedding so the model can act as a feature encoder for a downstream GNN.
    """

    def __init__(
        self,
        T=1500,
        n_classes=2,
        conv_ch=64,
        kernel_size_1=64,
        kernel_size_2=32,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_ch, kernel_size=kernel_size_1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),
            nn.GroupNorm(num_groups=4, num_channels=conv_ch),
            nn.Dropout(p=0.3),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=kernel_size_2, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),
            nn.GroupNorm(num_groups=8, num_channels=conv_ch),
            nn.Dropout(p=0.2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(conv_ch),
            nn.ReLU(inplace=True),
            nn.LayerNorm(conv_ch),
            nn.Dropout(p=0.3),
        )
        self.classifier = nn.Linear(conv_ch, n_classes)

    def forward(self, x, return_embedding=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        f = self.features(x)
        emb = self.head(f)
        out = self.classifier(emb)
        if return_embedding:
            return out, emb
        return out


# ----------------------------- training -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_class_weights(loader, device):
    """Run one pass over the loader to count class frequencies, return weights for CE."""
    counts = torch.zeros(2, dtype=torch.long)
    for _, labels, _, _ in loader:
        counts[0] += int((labels == 0).sum().item())
        counts[1] += int((labels == 1).sum().item())
    total = counts.sum().item()
    if counts.min().item() == 0:
        return None, counts.tolist()
    # Inverse-frequency weights, normalized so that mean weight = 1.
    w = total / (2.0 * counts.float())
    w = w / w.mean()
    return w.to(device), counts.tolist()


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_files = []
    with torch.no_grad():
        for sigs, labels, file_ids, file_paths in loader:
            sigs = sigs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(sigs)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            for fid in file_ids:
                all_files.append(file_paths[fid])
    avg_loss = total_loss / max(1, n_batches)
    preds = np.concatenate(all_preds) if all_preds else np.empty(0, dtype=np.int64)
    labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    return avg_loss, preds, labels, all_files


def main():
    p = argparse.ArgumentParser(description="Train a small 1D CNN per-electrode fibrosis classifier.")
    p.add_argument("--data-glob", default="AP_simulations_batch/*.npz")
    p.add_argument("--out-root", default="cnn_electrode_only")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-files", type=int, default=None,
                   help="Limit number of npz files for quick experiments.")
    p.add_argument("--fib-distance-threshold", type=float, default=1.5)
    p.add_argument("--T-max", type=int, default=1500)
    p.add_argument("--scaling", default="std", choices=["std", "min-max", "none"])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--files-per-batch", type=int, default=8,
                   help="Number of files per mini-batch (each file has up to ~100 electrodes).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--conv-ch", type=int, default=64)
    p.add_argument("--kernel-size-1", type=int, default=64)
    p.add_argument("--kernel-size-2", type=int, default=32)
    p.add_argument("--use-class-weights", action="store_true",
                   help="Use inverse-frequency class weights in CE loss.")
    p.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    p.set_defaults(use_class_weights=True)
    p.add_argument("--n-overlay-plots", type=int, default=5,
                   help="Number of random validation files to render as GT/pred overlays. Set 0 to skip.")
    p.add_argument("--overlay-seed", type=int, default=0,
                   help="RNG seed for picking the validation files to plot.")
    p.add_argument("--label-names", default="fibrosis",
                   choices=list(LABEL_NAME_PRESETS.keys()),
                   help="Patch type for labelling reports/plots: "
                        "'fibrosis' (lower D), 'higher_k' (heightened excitability via higher k), "
                        "'lower_a' (heightened excitability via lower a). "
                        "Affects only labels/strings — the binary task is unchanged.")
    args = p.parse_args()

    pos_name, neg_name = LABEL_NAME_PRESETS[args.label_names]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    file_paths = sorted(glob.glob(args.data_glob))
    if not file_paths:
        raise FileNotFoundError(f"No files matched: {args.data_glob}")
    if args.max_files is not None:
        file_paths = file_paths[: args.max_files]
    print(f"Found {len(file_paths)} npz files.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(file_paths))
    n_val = max(1, int(round(args.val_frac * len(file_paths))))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    full_ds = ElectrodeFileDataset(
        file_paths,
        T_max=args.T_max,
        fib_distance_threshold=args.fib_distance_threshold,
        scaling=args.scaling,
    )
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())
    print(f"Train files: {len(train_ds)} | Val files: {len(val_ds)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.files_per_batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_files,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.files_per_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=collate_files,
        persistent_workers=args.num_workers > 0,
    )

    if args.use_class_weights:
        print("Estimating class weights from training set...")
        class_weights, counts = estimate_class_weights(train_loader, device)
        print(f"Class counts (train): {neg_name}={counts[0]}, {pos_name}={counts[1]}")
        if class_weights is None:
            print("One class missing in train set; falling back to unweighted CE.")
        else:
            print(f"Class weights: {class_weights.detach().cpu().tolist()}")
    else:
        class_weights = None

    model = ElectrodeCNN(
        T=args.T_max,
        n_classes=2,
        conv_ch=args.conv_ch,
        kernel_size_1=args.kernel_size_1,
        kernel_size_2=args.kernel_size_2,
    ).to(device)
    # Materialize LazyLinear in the head so parameter counts and checkpoints are complete.
    with torch.no_grad():
        model(torch.zeros(1, args.T_max, device=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    run_name = args.run_name or (
        f"cnn_elec_{args.label_names}_thr{args.fib_distance_threshold:g}"
        f"_n{len(file_paths)}_seed{args.seed}"
    )
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "n_params": n_params, "device": str(device)}, f, indent=2)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for sigs, labels, _, _ in train_loader:
            sigs = sigs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(sigs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(1, n_batches)
        train_losses.append(train_loss)

        val_loss, _, _, _ = evaluate(model, val_loader, device, loss_fn)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "T": args.T_max,
                        "n_classes": 2,
                        "conv_ch": args.conv_ch,
                        "kernel_size_1": args.kernel_size_1,
                        "kernel_size_2": args.kernel_size_2,
                    },
                },
                out_dir / "best_model.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "val_loss": val_losses[-1] if val_losses else None,
            "model_state_dict": model.state_dict(),
        },
        out_dir / "final_model.pt",
    )

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title(f"Loss — {run_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "losses.png")
    plt.close()

    # Reload best model for final evaluation.
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_preds, val_labels, val_files = evaluate(model, val_loader, device, loss_fn)
    print(f"Best-epoch val loss: {val_loss:.4f}")

    report = classification_report(
        val_labels, val_preds, labels=[0, 1],
        target_names=[neg_name, pos_name], zero_division=0,
    )
    print(report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cbar=False,
        xticklabels=[neg_name, pos_name],
        yticklabels=[neg_name, pos_name],
    )
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Validation confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    np.savez(
        out_dir / "val_predictions.npz",
        preds=val_preds,
        labels=val_labels,
        files=np.array(val_files),
    )

    if args.n_overlay_plots > 0:
        written = save_random_val_overlays(
            out_dir, val_preds, val_labels, val_files,
            n=args.n_overlay_plots, seed=args.overlay_seed,
            pos_name=pos_name, neg_name=neg_name,
        )
        print(f"Wrote {len(written)} overlay plot(s) to {out_dir / 'val_overlays'}")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
