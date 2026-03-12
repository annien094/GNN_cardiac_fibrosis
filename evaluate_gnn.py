import glob
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize
from skimage.measure import label, regionprops


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sim_file(path):
    """
    NPZ-only loader.

    Returns:
      phie: (E, T)
      fiblocs: object array of (N,2) [row,col]
      elecpos: (E,2) in mm
      t: (T,)
      x_min, x_max, y_min, y_max
      meta: dict
    """
    d = np.load(path, allow_pickle=True)

    phie = np.asarray(d["phie"])      # (E, T)
    t = np.asarray(d["t_sav"]).squeeze()

    fiblocs = d["fiblocs"]
    elecpos = np.asarray(d["elecpos"], dtype=np.float64) / 10.0

    H0, W0 = 100, 100
    if elecpos.size > 0 and np.isfinite(elecpos).all():
        x_min, x_max = float(elecpos[:, 0].min()), float(elecpos[:, 0].max())
        y_min, y_max = float(elecpos[:, 1].min()), float(elecpos[:, 1].max())
        if abs(x_max - x_min) < 1e-12:
            x_min, x_max = 0.0, float(W0 - 1)
        if abs(y_max - y_min) < 1e-12:
            y_min, y_max = 0.0, float(H0 - 1)
    else:
        x_min, x_max = 0.0, float(W0 - 1)
        y_min, y_max = 0.0, float(H0 - 1)

    meta = {}
    for k in ["stim_center", "patch_type", "seed", "ncells", "D0", "Dfac", "npatches"]:
        if k in d:
            meta[k] = d[k]

    return phie, fiblocs, elecpos, t, x_min, x_max, y_min, y_max, meta


def build_gt_label_base(fiblocs, H0, W0, path_for_debug=""):
    """
    Builds GT label on native grid size (H0,W0).
    Supports:
      - rectangle list (old .mat): fiblocs shape (N,4) with [x_start, y_start, width, height] in MATLAB indexing
      - irregular coords (old irreg + new .npz): object array of arrays with [row,col] or [x,y] coords
    """
    gt = np.zeros((H0, W0), dtype=np.float32)

    is_object_like = (isinstance(fiblocs, np.ndarray) and fiblocs.dtype == object)

    if not is_object_like:
        fiblocs_arr = np.asarray(fiblocs)
        if fiblocs_arr.ndim == 2 and fiblocs_arr.shape[1] >= 4:
            for row in range(fiblocs_arr.shape[0]):
                x_start = int(fiblocs_arr[row, 0] - 1)
                y_start = int(fiblocs_arr[row, 1] - 1)
                width = int(fiblocs_arr[row, 2])
                height = int(fiblocs_arr[row, 3])

                x0 = max(0, x_start)
                y0 = max(0, y_start)
                x1 = min(W0, x_start + width)
                y1 = min(H0, y_start + height)
                if x1 > x0 and y1 > y0:
                    gt[y0:y1, x0:x1] = 1.0
        else:
            is_object_like = True

    if is_object_like:
        pieces = []
        flat_iter = fiblocs.flat if isinstance(fiblocs, np.ndarray) else [fiblocs]
        for a in flat_iter:
            if a is None:
                continue
            arr = np.asarray(a)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, 2)
            pieces.append(arr)

        if len(pieces) > 0:
            all_coords = np.concatenate(pieces, axis=0)
            rc0 = all_coords[:, 0]
            rc1 = all_coords[:, 1]

            one_indexed = (rc0.max() >= H0) or (rc1.max() >= W0) or (rc0.min() >= 1 and rc1.min() >= 1)
            if one_indexed:
                r = rc0.astype(int) - 1
                c = rc1.astype(int) - 1
            else:
                r = rc0.astype(int)
                c = rc1.astype(int)

            valid = (r >= 0) & (r < H0) & (c >= 0) & (c < W0)
            if not np.all(valid):
                print(
                    f"Warning: dropping {np.count_nonzero(~valid)} fib coords out of bounds in {path_for_debug}"
                )
            r = r[valid]
            c = c[valid]
            gt[r, c] = 1.0

    return gt


class Rodrigo_GNN_edge_features(nn.Module):
    def __init__(
        self,
        in_ch=1,
        gnn_type="GAT",
        out_ch=2,
        conv_ch=64,
        gnn_ch1=32,
        gnn_ch2=32,
        gnn_nlayer=2,
        edge_dim=1,
        kernel_size_1=64,
        kernel_size_2=32,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, conv_ch, kernel_size_1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),
            nn.GroupNorm(num_groups=4, num_channels=conv_ch),
            nn.Dropout(p=0.3),
            nn.Conv1d(conv_ch, conv_ch, kernel_size_2, padding="same"),
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
        self.gnn_type = gnn_type
        self.gcn = GCNConv(gnn_ch1, gnn_ch1)
        self.gat1 = GATv2Conv(conv_ch, gnn_ch1, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(gnn_ch1, gnn_ch2, edge_dim=edge_dim)
        self.classifier = nn.Linear(gnn_ch2, out_ch)
        self.gnn_nlayer = gnn_nlayer

    def forward(self, x, edge_index, edge_attr=None):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.head(x)

        if self.gnn_type == "GAT":
            h, a = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
            h = h.relu()
            h, a = self.gat2(h, edge_index, edge_attr, return_attention_weights=True)
            h = h.relu()
            out = self.classifier(h)
            return out, h, a

        if self.gnn_type == "GCN":
            for _ in range(self.gnn_nlayer):
                h = self.gcn(x, edge_index)
                h = h.relu()
                x = h
            out = self.classifier(h)
            return out, h

        raise ValueError(f"Unsupported gnn_type: {self.gnn_type}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_preprocessed_graph_paths(preprocessed_dir):
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed graph directory does not exist: {preprocessed_dir}")

    index_path = preprocessed_dir / "graph_index_6000_seed42.txt"
    if index_path.exists():
        with open(index_path, "r") as f:
            graph_paths = [Path(line.strip()) for line in f if line.strip()]
    else:
        graph_paths = sorted(preprocessed_dir.glob("*.pt"))

    if not graph_paths:
        raise FileNotFoundError(f"No preprocessed .pt graphs found in: {preprocessed_dir}")

    return graph_paths


class PreprocessedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_paths):
        self.graph_paths = list(graph_paths)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        graph_path = self.graph_paths[idx]
        return torch.load(graph_path, map_location="cpu")


def evaluate_on_validation(
    model,
    val_set,
    out_dir,
    target_shape=(30, 30),
    dim=2,
    savefig=True,
    error_threshold=0.05,
    do_reports=True,
    do_plots=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()

    failure_summary = []
    failure_details = []
    success_details = []

    for i in range(len(val_set)):
        sample_graph = val_set[i].clone()
        with torch.no_grad():
            g = sample_graph.to(device)
            output = model(g.x, g.edge_index, getattr(g, "edge_attr", None))
            logits = output[0] if isinstance(output, (tuple, list)) else output
            preds = logits.argmax(dim=1)

        tm_mask = sample_graph.tm_mask
        y_true = sample_graph.y[tm_mask].cpu().numpy()
        y_pred = preds[tm_mask].cpu().numpy()

        failures, all_patches = analyze_fibrotic_patch_failures(
            y_true,
            y_pred,
            target_shape,
            error_threshold=error_threshold,
        )
        successes = [p for p in all_patches if p["error_rate"] < error_threshold]
        failure_summary.append(
            {
                "sample_index": i,
                "num_patches": len(all_patches),
                "num_failed_patches": len(failures),
                "failure_rate": (len(failures) / max(1, len(all_patches))),
            }
        )
        failure_details.append(
            {
                "sample_index": i,
                "failures": failures,
            }
        )
        success_details.append(
            {
                "sample_index": i,
                "successes": successes,
            }
        )

        if do_reports:
            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["non-fibrotic", "fibrotic"],
            )
            print(report)
            with open(out_dir / f"classification_report_{i}.txt", "w") as f:
                f.write(report)

        if do_plots:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cbar=False,
                xticklabels=["non-fibrotic", "fibrotic"],
                yticklabels=["non-fibrotic", "fibrotic"],
                annot_kws={"size": 20},
            )
            plt.xlabel("Predicted", fontsize=14)
            plt.ylabel("True", fontsize=14)
            plt.title(f"Validation Confusion Matrix {i}", fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            if savefig:
                plt.savefig(out_dir / f"validation_confusion_matrix_{i}.png")
            plt.show()
            plt.close()

        if do_plots:
            if target_shape is None:
                target_shape = (100, 100)
            if dim == 2:
                plt.figure(figsize=(6, 5))
                elec_coords = sample_graph.elec_coords
                if elec_coords.size > 0 and np.isfinite(elec_coords).all():
                    x_min, x_max = float(elec_coords[:, 0].min()), float(elec_coords[:, 0].max())
                    y_min, y_max = float(elec_coords[:, 1].min()), float(elec_coords[:, 1].max())
                else:
                    x_min, x_max = 0.0, 10.0
                    y_min, y_max = 0.0, 10.0
                plt.imshow(
                    y_pred.reshape(target_shape),
                    extent=[x_min, x_max, y_min, y_max],
                    cmap="gray",
                    origin="lower",
                    aspect="equal",
                )
                plt.colorbar(ticks=[0, 1])
                x_phys = np.linspace(x_min, x_max, target_shape[1])
                y_phys = np.linspace(y_min, y_max, target_shape[0])
                Xg, Yg = np.meshgrid(x_phys, y_phys, indexing="xy")
                plt.contour(
                    Xg,
                    Yg,
                    y_true.reshape(target_shape),
                    levels=[0.5],
                    colors="red",
                    linewidths=1.5,
                )
                plt.scatter(elec_coords[:, 0], elec_coords[:, 1], c="red", label="Electrodes")
                meta = getattr(sample_graph, "meta", {}) or {}
                stim_center = meta.get("stim_center") if isinstance(meta, dict) else None
                if stim_center is not None:
                    stim_center = np.asarray(stim_center).reshape(-1)
                    if stim_center.size >= 2:
                        sx, sy = float(stim_center[0]), float(stim_center[1])
                        H0, W0 = 100, 100
                        looks_like_grid = (
                            (sx > x_max or sy > y_max)
                            and (0.0 <= sx <= W0)
                            and (0.0 <= sy <= H0)
                        )
                        if looks_like_grid:
                            one_indexed = (sx >= 1.0 and sy >= 1.0)
                            if one_indexed:
                                sx -= 1.0
                                sy -= 1.0
                            sx = x_min + (sx / max(1.0, W0 - 1)) * (x_max - x_min)
                            sy = y_min + (sy / max(1.0, H0 - 1)) * (y_max - y_min)
                        plt.scatter([sx], [sy], c="cyan", marker="*", s=120, label="Pacing")
                plt.title(f"Predicted Fibrotic Region with GT contours - validation {i}")
                plt.tight_layout()
                if savefig:
                    plt.savefig(out_dir / f"pred_vs_gt_{i}_w_elecs.png")
                plt.show()
                plt.close()
            elif dim == 3:
                raise NotImplementedError("3D plotting not implemented in this evaluation script.")

    summary_path = out_dir / "fibrosis_patch_failure_summary.txt"
    failed_props_path = out_dir / "fibrosis_patch_failed_properties.txt"
    success_props_path = out_dir / "fibrosis_patch_success_properties.txt"
    total_patches = sum(s["num_patches"] for s in failure_summary)
    total_failed = sum(s["num_failed_patches"] for s in failure_summary)
    overall_rate = total_failed / max(1, total_patches)
    with open(summary_path, "w") as f:
        f.write("Fibrotic patch failure summary\n")
        f.write(f"Error threshold: {error_threshold:.3f}\n")
        f.write(f"Total patches: {total_patches}\n")
        f.write(f"Failed patches: {total_failed}\n")
        f.write(f"Overall failure rate: {overall_rate:.4f}\n\n")
        f.write("Per-sample stats (index, num_patches, num_failed_patches, failure_rate):\n")
        for s in failure_summary:
            f.write(
                f"{s['sample_index']}\t{s['num_patches']}\t{s['num_failed_patches']}\t{s['failure_rate']:.4f}\n"
            )
        f.write("\nFailed patch properties:\n")
        f.write(
            "Fields: sample_index, label, area, perimeter, eccentricity, solidity, extent, "
            "major_axis_length, minor_axis_length, centroid_rc, error_rate, misclassified\n"
        )
        for entry in failure_details:
            sample_index = entry["sample_index"]
            for info in entry["failures"]:
                f.write(
                    f"{sample_index}\t{info['label']}\t{info['area']}\t"
                    f"{info['perimeter']:.4f}\t{info['eccentricity']:.4f}\t"
                    f"{info['solidity']:.4f}\t{info['extent']:.4f}\t"
                    f"{info['major_axis_length']:.4f}\t{info['minor_axis_length']:.4f}\t"
                    f"({info['centroid_rc'][0]:.2f},{info['centroid_rc'][1]:.2f})\t"
                    f"{info['error_rate']:.4f}\t{info['misclassified']}\n"
                )
        f.write("\nSuccessful patch properties:\n")
        f.write(
            "Fields: sample_index, label, area, perimeter, eccentricity, solidity, extent, "
            "major_axis_length, minor_axis_length, centroid_rc, error_rate, misclassified\n"
        )
        for entry in success_details:
            sample_index = entry["sample_index"]
            for info in entry["successes"]:
                f.write(
                    f"{sample_index}\t{info['label']}\t{info['area']}\t"
                    f"{info['perimeter']:.4f}\t{info['eccentricity']:.4f}\t"
                    f"{info['solidity']:.4f}\t{info['extent']:.4f}\t"
                    f"{info['major_axis_length']:.4f}\t{info['minor_axis_length']:.4f}\t"
                    f"({info['centroid_rc'][0]:.2f},{info['centroid_rc'][1]:.2f})\t"
                    f"{info['error_rate']:.4f}\t{info['misclassified']}\n"
                )

    with open(failed_props_path, "w") as f:
        f.write(
            "Fields: sample_index, label, area, perimeter, eccentricity, solidity, extent, "
            "major_axis_length, minor_axis_length, centroid_rc, error_rate, misclassified\n"
        )
        for entry in failure_details:
            sample_index = entry["sample_index"]
            for info in entry["failures"]:
                f.write(
                    f"{sample_index}\t{info['label']}\t{info['area']}\t"
                    f"{info['perimeter']:.4f}\t{info['eccentricity']:.4f}\t"
                    f"{info['solidity']:.4f}\t{info['extent']:.4f}\t"
                    f"{info['major_axis_length']:.4f}\t{info['minor_axis_length']:.4f}\t"
                    f"({info['centroid_rc'][0]:.2f},{info['centroid_rc'][1]:.2f})\t"
                    f"{info['error_rate']:.4f}\t{info['misclassified']}\n"
                )

    with open(success_props_path, "w") as f:
        f.write(
            "Fields: sample_index, label, area, perimeter, eccentricity, solidity, extent, "
            "major_axis_length, minor_axis_length, centroid_rc, error_rate, misclassified\n"
        )
        for entry in success_details:
            sample_index = entry["sample_index"]
            for info in entry["successes"]:
                f.write(
                    f"{sample_index}\t{info['label']}\t{info['area']}\t"
                    f"{info['perimeter']:.4f}\t{info['eccentricity']:.4f}\t"
                    f"{info['solidity']:.4f}\t{info['extent']:.4f}\t"
                    f"{info['major_axis_length']:.4f}\t{info['minor_axis_length']:.4f}\t"
                    f"({info['centroid_rc'][0]:.2f},{info['centroid_rc'][1]:.2f})\t"
                    f"{info['error_rate']:.4f}\t{info['misclassified']}\n"
                )


def analyze_fibrotic_patch_failures(
    y_true,
    y_pred,
    target_shape,
    error_threshold=0.05,
):
    """
    Identify fibrotic patches whose misclassification rate is >= error_threshold.

    Args:
        y_true: 1D or 2D array of GT labels (0/1).
        y_pred: 1D or 2D array of predicted labels (0/1).
        target_shape: (H, W) used to reshape 1D arrays.
        error_threshold: fraction of pixels in a patch misclassified to flag as failure.

    Returns:
        failures: list of dicts with region properties and error stats.
        all_patches: list of dicts for all fibrotic patches.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(target_shape)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(target_shape)

    gt_mask = y_true_arr.astype(bool)
    pred_mask = y_pred_arr.astype(bool)

    labeled = label(gt_mask)
    failures = []
    all_patches = []

    for region in regionprops(labeled):
        region_mask = labeled == region.label
        area = int(region.area)
        if area <= 0:
            continue
        misclassified = int(np.count_nonzero(region_mask & (pred_mask != gt_mask)))
        error_rate = misclassified / float(area)

        info = {
            "label": int(region.label),
            "area": area,
            "perimeter": float(region.perimeter),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "extent": float(region.extent),
            "major_axis_length": float(region.major_axis_length),
            "minor_axis_length": float(region.minor_axis_length),
            "centroid_rc": (float(region.centroid[0]), float(region.centroid[1])),
            "error_rate": float(error_rate),
            "misclassified": misclassified,
        }
        all_patches.append(info)
        if error_rate >= error_threshold:
            failures.append(info)

    return failures, all_patches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved GNN model on validation set.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Folder containing the saved .pt model checkpoint.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="preprocessed_graphs",
        help="Folder containing preprocessed graph .pt files.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Evaluate only the first N graphs (by sorted path order).",
    )
    parser.add_argument(
        "--shape-only",
        action="store_true",
        help="Run only shape failure analysis (skip reports and plots).",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip classification reports.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip confusion matrix and prediction/GT plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    model_files = sorted(model_dir.glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in: {model_dir}")
    final_models = [p for p in model_files if "final_model" in p.name]
    if not final_models:
        raise FileNotFoundError(
            f"No model files containing 'final_model' found in: {model_dir}"
        )
    model_file = final_models[-1]

    out_dir = model_dir / "best_model_results"

    graph_paths = load_preprocessed_graph_paths(args.preprocessed_dir)
    if args.max_graphs is not None:
        if args.max_graphs <= 0:
            raise ValueError("--max-graphs must be a positive integer.")
        graph_paths = graph_paths[: args.max_graphs]
    dataset = PreprocessedGraphDataset(graph_paths)
    if len(dataset) == 0:
        raise ValueError("No preprocessed graphs found; aborting evaluation.")

    first_graph = dataset[0]
    num_tm = int(first_graph.tm_mask.sum().item())
    side = int(round(np.sqrt(num_tm)))
    if side * side != num_tm:
        print("Warning: tm_mask size is not a perfect square; using a flattened shape for plotting.")
        target_shape = (num_tm, 1)
    else:
        target_shape = (side, side)


    # n_total = len(dataset)
    # n_train = max(1, int(0.8 * n_total))
    # n_val = max(1, n_total - n_train)
    # train_set, val_set = torch.utils.data.random_split(
    #     dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    # )
    # Evaluate all graphs listed in graph_index.txt (no split).
    val_set = dataset

    model = Rodrigo_GNN_edge_features(
        in_ch=1,
        out_ch=2,
        gnn_type="GAT",
        gnn_nlayer=2,
        edge_dim=1,
    ).to(device)

    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    do_reports = not args.no_reports
    do_plots = not args.no_plots
    if args.shape_only:
        do_reports = False
        do_plots = False

    evaluate_on_validation(
        model,
        val_set,
        out_dir=out_dir,
        target_shape=target_shape,
        dim=2,
        savefig=True,
        error_threshold=0.95,
        do_reports=do_reports,
        do_plots=do_plots,
    )
