import argparse
import glob
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.transform import resize
from pathlib import Path


def load_sim_file(path):
    d = np.load(path, allow_pickle=True)

    phie = np.asarray(d["phie"])
    t = np.asarray(d["t_sav"]).squeeze()
    fiblocs = d["fiblocs"]
    elecpos = np.asarray(d["elecpos"], dtype=np.float64) / 10.0

    H0 = 100
    W0 = 100
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


def preprocess_and_save_graphs(
    file_list,
    out_dir,
    target_shape,
    t_trunc_start,
    t_trunc_end,
    graph_type,
    delta,
    eps,
    k_nearest,
    scaling,
    task,
    noise,
    sparse_phie,
    T_max,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "graph_index.txt"
    saved_paths = []

    for f in file_list:
        try:
            phie, fiblocs, electrode_locs, t, x_min, x_max, y_min, y_max, meta = load_sim_file(f)
        except Exception as e:
            print(f"Skipping {f} due to load error: {e}")
            continue

        print(f"Creating graph data from file {f}.")

        H0, W0, T0 = 100, 100, phie.shape[1]

        gt_label_base = build_gt_label_base(fiblocs, H0, W0, path_for_debug=f)

        if target_shape:
            gt_down = resize(gt_label_base, target_shape, order=0, preserve_range=True, anti_aliasing=False)

            gt_classes = (gt_down > 0.5).astype(np.int64).reshape(-1)

            if task == "regression":
                gt_values = np.array([0.01, 0.1], dtype=np.float32)
                gt_label = gt_values[gt_classes].reshape(-1)
            else:
                gt_label = gt_classes

            xg = np.linspace(x_min, x_max, target_shape[1])
            yg = np.linspace(y_min, y_max, target_shape[0])
        else:
            xg = np.linspace(x_min, x_max, W0)
            yg = np.linspace(y_min, y_max, H0)
            gt_label = gt_label_base.astype(np.int64).reshape(-1)

        phie_tr = phie[:, t_trunc_start:t_trunc_end]
        Tprime = phie_tr.shape[1]

        elec_coords = electrode_locs.astype(np.float64)
        if sparse_phie:
            keep_idx = np.random.choice(phie_tr.shape[0], int(phie_tr.shape[0] * sparse_phie), replace=False)
            phie_tr = phie_tr[keep_idx, :]
            elec_coords = elec_coords[keep_idx, :]

        H, W = target_shape if target_shape else (H0, W0)

        XX, YY = np.meshgrid(xg, yg)
        tm_coords = np.column_stack([XX.ravel(), YY.ravel()])
        tm_coords_t = torch.tensor(tm_coords, dtype=torch.float)

        tm_xy_min = tm_coords_t.min(dim=0, keepdim=True).values
        tm_xy_max = tm_coords_t.max(dim=0, keepdim=True).values
        tm_xy = (tm_coords_t - tm_xy_min) / (tm_xy_max - tm_xy_min).clamp_min(1e-12)

        x_tm = torch.zeros((tm_xy.shape[0], Tprime), dtype=torch.float)

        x_elec = torch.tensor(phie_tr, dtype=torch.float)
        x = torch.cat([x_elec, x_tm], dim=0)

        eps_scale = 1e-6
        if scaling == "min-max":
            mins = x.min(dim=1, keepdim=True).values
            maxs = x.max(dim=1, keepdim=True).values
            denom = (maxs - mins).clamp_min(eps_scale)
            x_scaled = (x - mins) / denom
        else:
            means = x.mean(dim=1, keepdim=True)
            stds = x.std(dim=1, keepdim=True)
            x_scaled = (x - means) / stds.clamp_min(eps_scale)

        if noise:
            x_scaled = x_scaled + noise * torch.randn_like(x_scaled)

        def pad_or_trim(x_in, T=T_max):
            L = x_in.shape[-1]
            if L == T:
                return x_in
            if L > T:
                return x_in[..., :T]
            pad_len = T - L
            return torch.nn.functional.pad(x_in, (0, pad_len))

        x_scaled_padded = pad_or_trim(x_scaled, T_max)

        num_tm = tm_coords.shape[0]
        num_elec = elec_coords.shape[0]
        num_nodes = num_elec + num_tm

        def build_elec_tm_edges():
            diff = tm_coords[:, None, :] - elec_coords[None, :, :]
            distance_matrix = np.linalg.norm(diff, axis=-1)

            mask = np.ones_like(distance_matrix, dtype=bool)
            if k_nearest is not None and k_nearest > 0:
                k = min(k_nearest, distance_matrix.shape[1])
                nn_idx = np.argpartition(distance_matrix, kth=k - 1, axis=1)[:, :k]
                mask = np.zeros_like(distance_matrix, dtype=bool)
                row_idx = np.arange(distance_matrix.shape[0])[:, None]
                mask[row_idx, nn_idx] = True

            if delta is not None:
                mask &= distance_matrix <= delta

            W_e = 1.0 / (distance_matrix + eps)
            W_e[~mask] = 0.0

            rows, cols = np.nonzero(W_e)
            weights = W_e[rows, cols].astype(np.float32)

            src = torch.from_numpy(cols).long()
            dst = torch.from_numpy(rows + num_elec).long()
            edge_index_local = torch.stack([src, dst], dim=0)
            edge_attr_local = torch.from_numpy(weights)
            return edge_index_local, edge_attr_local

        if graph_type == "fc-bipar":
            rows, cols = np.indices((num_tm, num_elec))
            rows = rows.flatten()
            cols = cols.flatten()
            src = torch.from_numpy(cols).long()
            dst = torch.from_numpy(rows + num_elec).long()
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = None

        elif graph_type == "dist-based":
            edge_index, edge_attr = build_elec_tm_edges()

        elif graph_type == "dist-based-add-tm":
            edge_index, edge_attr = build_elec_tm_edges()

            delta_tm = 1.0
            eps_tm = eps

            tm_diff = tm_coords[:, None, :] - tm_coords[None, :, :]
            tm_dist = np.linalg.norm(tm_diff, axis=-1)

            W_tm = 1.0 / (tm_dist + eps_tm)
            W_tm[tm_dist > delta_tm] = 0.0
            np.fill_diagonal(W_tm, 0.0)

            rows_tm, cols_tm = np.nonzero(W_tm)
            weights_tm = W_tm[rows_tm, cols_tm].astype(np.float32)

            src_tm = torch.from_numpy(rows_tm + num_elec).long()
            dst_tm = torch.from_numpy(cols_tm + num_elec).long()

            edge_index_tm = torch.stack([src_tm, dst_tm], dim=0)
            edge_index = torch.cat([edge_index, edge_index_tm], dim=1)
            edge_attr = torch.cat([edge_attr, torch.from_numpy(weights_tm)], dim=0)

        if task == "regression":
            y_nodes = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
            y_nodes[num_elec:] = torch.from_numpy(gt_label).float()
        else:
            y_nodes = torch.full((num_nodes,), -100, dtype=torch.long)
            y_nodes[num_elec:] = torch.from_numpy(gt_label).long()

        tm_mask = torch.zeros(num_nodes, dtype=torch.bool)
        tm_mask[num_elec:] = True

        data_graph = Data(
            x=x_scaled_padded,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_nodes,
            tm_mask=tm_mask,
            num_elec=num_elec,
            pacing_info=f,
        )
        data_graph["elec_coords"] = elec_coords
        data_graph["meta"] = meta

        graph_path = out_dir / (Path(f).stem + ".pt")
        torch.save(data_graph, graph_path)
        saved_paths.append(graph_path)

    with open(index_path, "w") as f:
        for p in saved_paths:
            f.write(str(p) + "\n")

    print(f"Saved {len(saved_paths)} graphs to {out_dir}.")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Preprocess NPZ simulations into graph .pt files.")
    parser.add_argument("--file-pattern", default="AP_simulations_batch/*.npz")
    parser.add_argument("--out-dir", default="preprocessed_graphs")
    parser.add_argument("--target-h", type=int, default=30)
    parser.add_argument("--target-w", type=int, default=30)
    parser.add_argument("--t-trunc-start", type=int, default=0)
    parser.add_argument("--t-trunc-end", type=int, default=None)
    parser.add_argument("--graph-type", default="dist-based-add-tm")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--k-nearest", type=int, default=4)
    parser.add_argument("--scaling", default="std")
    parser.add_argument("--task", default="classification")
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--sparse-phie", type=float, default=None)
    parser.add_argument("--t-max", type=int, default=1500)
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.file_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files matched pattern: {args.file_pattern}")

    target_shape = (args.target_h, args.target_w) if args.target_h and args.target_w else None

    preprocess_and_save_graphs(
        file_list=file_list,
        out_dir=args.out_dir,
        target_shape=target_shape,
        t_trunc_start=args.t_trunc_start,
        t_trunc_end=args.t_trunc_end,
        graph_type=args.graph_type,
        delta=args.delta,
        eps=args.eps,
        k_nearest=args.k_nearest,
        scaling=args.scaling,
        task=args.task,
        noise=args.noise,
        sparse_phie=args.sparse_phie,
        T_max=args.t_max,
    )


if __name__ == "__main__":
    main()
