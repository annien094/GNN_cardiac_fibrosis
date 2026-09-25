"""
surface_to_graph.py

Turn curved-surface Aliev-Panfilov simulations (produced by
solveAP_surface_fenicsx_phie.py) into GNN graphs whose schema matches
preprocess_graphs.py, so the CNN+GAT model trained on the flat 2D data can be
applied directly (see eval_surface_gnn.py).

What's different from preprocess_graphs.py (flat 30x30 grid):
  * Nodes live on an ellipsoid patch, not a square grid. The dense FEniCSx mesh
    (~20k P1 nodes) is DOWNSAMPLED with KMeans on the 3D node coordinates to
    target_n ~= electrode_to_mesh_ratio * num_elec mesh nodes (default 9:1).
  * Fibrotic (low-D) nodes are a small minority, so KMeans clusters that contain
    any fibrotic node elect a fibrotic representative (``--fib-bias``, on by
    default). This force-includes low-D nodes during downsampling; each kept
    node's label is then read straight off its own D value.
  * The model's interp/nearest TM-init rebuilds a square grid from the electrode
    bbox and ignores real coordinates -- wrong for a scattered surface. We store
    the *real* per-node surface parameters as ``tm_coords`` (theta, phi) and the
    electrode parameters as ``elec_coords`` (theta, phi); eval_surface_gnn.py
    patches the interpolation to use them.

Everything else (electrodes-first node ordering, std scaling, T_max padding,
elec->TM kNN inverse-distance edges + TM-TM edges, y in {-100, 0, 1}, tm_mask,
num_elec) mirrors preprocess_graphs.py so the saved .pt + graph_index.txt drop
straight into train_gnn_with_pretrained_cnn.py / eval_surface_gnn.py.

Usage:
    python surface_to_graph.py \
        --file-pattern "sim_results/split_a_b/phie_surface_*.npz" \
        --out-dir surface_graphs \
        --electrode-to-mesh-ratio 9 \
        --coord-scale 0.627 \
        --tm-tm-scale 0.914
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


# ----------------------------------------------------------------- io

def ellipsoid_inverse_params(coords, rx, ry, rz):
    """Recover (theta, phi) for points on the ellipsoid (inverse of the map in
    solveAP_surface_fenicsx_phie.py): theta = arccos(z/rz), phi = atan2(y/ry, x/rx).
    """
    coords = np.asarray(coords, dtype=np.float64)
    theta = np.arccos(np.clip(coords[:, 2] / rz, -1.0, 1.0))
    phi = np.arctan2(coords[:, 1] / ry, coords[:, 0] / rx)
    return np.column_stack([theta, phi])


def load_surface_sim(path):
    """Load a curved-surface phie .npz. Returns a dict with everything the graph
    builder needs. Requires the extended npz written by the patched solver
    (node_coords, D_nodal, elec_thetaphi, rx/ry/rz, ...).
    """
    d = np.load(path, allow_pickle=True)
    required = ["phie", "electrodes", "node_coords", "D_nodal"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}. Re-run with the patched "
            "solveAP_surface_fenicsx_phie.py (it now saves mesh coords + D)."
        )

    phie = np.asarray(d["phie"], dtype=np.float64)          # (T, E)
    electrodes = np.asarray(d["electrodes"], dtype=np.float64)   # (E, 3)
    node_coords = np.asarray(d["node_coords"], dtype=np.float64)  # (N, 3)
    D_nodal = np.asarray(d["D_nodal"], dtype=np.float64).reshape(-1)  # (N,)

    rx = float(d["rx"]) if "rx" in d else 1.0
    ry = float(d["ry"]) if "ry" in d else 1.0
    rz = float(d["rz"]) if "rz" in d else 1.0

    # Electrode (theta, phi): use the stored params if present, else invert.
    if "elec_thetaphi" in d:
        elec_tp = np.asarray(d["elec_thetaphi"], dtype=np.float64)
    else:
        elec_tp = ellipsoid_inverse_params(electrodes, rx, ry, rz)

    D0 = float(d["D0"]) if "D0" in d else float(np.max(D_nodal))
    fib_factor = float(d["fib_factor"]) if "fib_factor" in d else 0.1
    # A node is fibrotic iff its D sits below the midpoint between healthy D0 and
    # the reduced D0*fib_factor -- robust to either crisp or lightly-smoothed D.
    fib_threshold = 0.5 * (D0 + D0 * fib_factor)

    # Abnormal-node label. Excitability sims (neg-a / higher-k) leave D uniform, so
    # the ground truth is the patch mask saved as label_nodal; fibrosis sims have
    # no label_nodal and are labelled by the low-D rule above. Downstream code
    # (fib_bias oversampling, tm_label, y) consumes this single generic mask.
    if "label_nodal" in d:
        abnormal = np.asarray(d["label_nodal"]).reshape(-1).astype(bool)
    else:
        abnormal = D_nodal < fib_threshold

    t = np.asarray(d["t"]).reshape(-1) if "t" in d else np.arange(phie.shape[0])

    return {
        "phie": phie,
        "electrodes": electrodes,
        "elec_thetaphi": elec_tp,
        "node_coords": node_coords,
        "node_thetaphi": ellipsoid_inverse_params(node_coords, rx, ry, rz),
        "D_nodal": D_nodal,
        "abnormal": abnormal,
        "fib_threshold": fib_threshold,
        "D0": D0,
        "fib_factor": fib_factor,
        "rx": rx, "ry": ry, "rz": rz,
        "t": t,
        "path": str(path),
    }


# ----------------------------------------------------------- downsampling

def kmeans_downsample(node_coords, is_fib, target_n, fib_bias=True, seed=0):
    """Downsample dense mesh nodes to ~target_n representatives via KMeans on the
    3D node coordinates (roughly uniform surface coverage).

    For each cluster the representative is the node nearest the cluster centroid.
    When ``fib_bias`` and the cluster contains any fibrotic (low-D) node, the
    representative is instead the fibrotic node nearest the centroid -- this is
    the "oversample fibrotic" rule: every cluster that touches the patch keeps a
    fibrotic node, so small patches survive downsampling. Labels are then read
    off each representative's own D value (handled by the caller).

    Returns kept_idx (target_n indices into node_coords).
    """
    from sklearn.cluster import MiniBatchKMeans

    n = node_coords.shape[0]
    k = int(min(target_n, n))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                         batch_size=max(256, 3 * k))
    labels = km.fit_predict(node_coords)
    centroids = km.cluster_centers_

    kept = np.full(k, -1, dtype=np.int64)
    d2_to_centroid = np.sum((node_coords - centroids[labels]) ** 2, axis=1)

    for c in range(k):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        if fib_bias and np.any(is_fib[members]):
            members = members[is_fib[members]]    # restrict to fibrotic members
        kept[c] = members[np.argmin(d2_to_centroid[members])]

    return kept[kept >= 0]


# ------------------------------------------------------------- edges

def _knn_inverse_distance(src_xy, dst_xy, k, w_max, eps):
    """k-nearest-neighbour edges from each dst point to its k nearest src points,
    weighted by 1/(distance + eps) (capped at w_max). Returns (rows, cols, w):
    rows index dst, cols index src.
    """
    diff = dst_xy[:, None, :] - src_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)                # (n_dst, n_src)
    kk = max(1, min(int(k), dist.shape[1]))
    nn_idx = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    row_idx = np.arange(dist.shape[0])[:, None]
    mask = np.zeros_like(dist, dtype=bool)
    mask[row_idx, nn_idx] = True
    W = 1.0 / (dist + eps)
    if w_max is not None:
        W = np.minimum(W, w_max)
    W[~mask] = 0.0
    rows, cols = np.nonzero(W)
    return rows, cols, W[rows, cols].astype(np.float32)


def build_edges(elec_xy, tm_xy, num_elec, k_elec, k_tm, w_max, eps,
                add_tm_tm=True, tm_tm_scale=1.0):
    """Bipartite electrode->TM kNN edges (+ optional TM-TM kNN edges), in the
    same 2D coordinate space used for interpolation. Mirrors the
    "dist-based-add-tm" graph in preprocess_graphs.py: src = electrode index,
    dst = TM index + num_elec.

    ``tm_tm_scale`` multiplies ONLY the TM-TM edge weights, decoupling their
    magnitude from elec->TM. A single coord_scale preserves the elec->TM : TM-TM
    ratio, which differs from the flat grid (electrode/TM density differs on the
    surface); use this to match both flat edge_attr targets at once and avoid
    over-smoothing.
    """
    rows, cols, w = _knn_inverse_distance(elec_xy, tm_xy, k_elec, w_max, eps)
    src = torch.from_numpy(cols).long()                 # electrode
    dst = torch.from_numpy(rows + num_elec).long()      # TM (offset)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.from_numpy(w)

    if add_tm_tm and k_tm and k_tm > 0:
        r2, c2, w2 = _knn_inverse_distance(tm_xy, tm_xy, k_tm + 1, w_max, eps)
        keep = r2 != c2                                 # drop self-loops
        r2, c2, w2 = r2[keep], c2[keep], w2[keep]
        w2 = (w2 * tm_tm_scale).astype(np.float32)      # independent TM-TM magnitude
        src_tm = torch.from_numpy(c2 + num_elec).long()
        dst_tm = torch.from_numpy(r2 + num_elec).long()
        edge_index = torch.cat([edge_index, torch.stack([src_tm, dst_tm], 0)], dim=1)
        edge_attr = torch.cat([edge_attr, torch.from_numpy(w2)], dim=0)

    return edge_index, edge_attr


# ------------------------------------------------------------- build one graph

def build_graph(sim, *, electrode_to_mesh_ratio, t_trunc_start, t_trunc_end,
                k_elec, k_tm, w_max, eps, coord_scale, scaling, T_max,
                fib_bias, add_tm_tm, seed, edge_space="thetaphi",
                tm_tm_scale=1.0):
    phie = sim["phie"]                                  # (T, E)
    num_elec = phie.shape[1]
    target_n = int(round(electrode_to_mesh_ratio * num_elec))

    is_fib = sim["abnormal"]                            # (N,) generic abnormal mask
    kept = kmeans_downsample(sim["node_coords"], is_fib, target_n,
                             fib_bias=fib_bias, seed=seed)
    num_tm = kept.shape[0]

    tm_label = is_fib[kept].astype(np.int64)            # 0/1 per kept node
    tm_tp = sim["node_thetaphi"][kept] * coord_scale    # (num_tm, 2) interp coords
    tm_xyz = sim["node_coords"][kept]                   # (num_tm, 3)
    elec_tp = sim["elec_thetaphi"] * coord_scale        # (num_elec, 2)

    # --- node features: electrodes carry phie traces, TM nodes start at zeros ---
    phie_eT = phie.T                                    # (E, T)
    phie_tr = phie_eT[:, t_trunc_start:t_trunc_end]
    Tprime = phie_tr.shape[1]

    x_elec = torch.tensor(phie_tr, dtype=torch.float)
    x_tm = torch.zeros((num_tm, Tprime), dtype=torch.float)

    eps_scale = 1e-6
    if scaling == "min-max":
        e_min, e_max = x_elec.min(), x_elec.max()
        x_elec_s = (x_elec - e_min) / (e_max - e_min).clamp_min(eps_scale)
        t_min, t_max = x_tm.min(), x_tm.max()
        x_tm_s = (x_tm - t_min) / (t_max - t_min).clamp_min(eps_scale)
    else:  # "std" -- per-block global standardisation (matches preprocess_graphs)
        x_elec_s = (x_elec - x_elec.mean()) / x_elec.std().clamp_min(eps_scale)
        x_tm_s = (x_tm - x_tm.mean()) / x_tm.std().clamp_min(eps_scale)
    x_scaled = torch.cat([x_elec_s, x_tm_s], dim=0)

    def pad_or_trim(x_in, T=T_max):
        L = x_in.shape[-1]
        if L == T:
            return x_in
        if L > T:
            return x_in[..., :T]
        return torch.nn.functional.pad(x_in, (0, T - L))

    x_scaled = pad_or_trim(x_scaled, T_max)

    # --- edges (electrode->TM kNN inverse-distance, optional TM-TM kNN) ---
    # Pick the coordinate space the kNN / inverse-distance weights live in.
    # "thetaphi" reuses the (theta,phi) interp coords, but that metric is
    # distorted on an ellipsoid (phi wraps at +-pi, distances near the poles are
    # badly non-uniform), so the neighbourhood graph doesn't match the
    # local-Euclidean structure the flat model was trained on. "xyz" builds edges
    # from the real 3D chord distance instead -- far more faithful surface
    # proximity. coord_scale still applies (re-tune it for xyz; the (theta,phi)
    # value won't carry over since the distance units differ).
    if edge_space == "xyz":
        elec_edge = sim["electrodes"] * coord_scale          # (num_elec, 3)
        tm_edge = sim["node_coords"][kept] * coord_scale     # (num_tm, 3)
    else:  # "thetaphi"
        elec_edge = elec_tp
        tm_edge = tm_tp
    edge_index, edge_attr = build_edges(
        elec_edge, tm_edge, num_elec, k_elec=k_elec, k_tm=k_tm,
        w_max=w_max, eps=eps, add_tm_tm=add_tm_tm, tm_tm_scale=tm_tm_scale)

    # Quick edge_attr health check against the flat-training graphs
    # (elec->TM median ~0.5745, TM-TM median ~1.554 in preprocess_graphs.py):
    src_idx = edge_index[0].numpy()
    ew = edge_attr.numpy()
    e2t = ew[src_idx < num_elec]
    t2t = ew[src_idx >= num_elec]
    suggest = coord_scale * (np.median(e2t) / 0.5745) if e2t.size else float("nan")
    # tm_tm_scale that would drag the (already tm_tm_scale'd) TM-TM median onto flat:
    suggest_tmtm = (tm_tm_scale * (1.554 / np.median(t2t))
                    if t2t.size and np.median(t2t) > 0 else float("nan"))
    print(f"    edge_attr[{edge_space}] medians: elec->TM "
          f"{np.median(e2t):.4f} (flat 0.5745), TM-TM "
          f"{(np.median(t2t) if t2t.size else float('nan')):.4f} (flat 1.554); "
          f"coord_scale->{suggest:.3g} matches flat elec->TM, "
          f"tm_tm_scale->{suggest_tmtm:.3g} matches flat TM-TM")

    # --- labels / masks (electrodes ignored via -100) ---
    num_nodes = num_elec + num_tm
    y = torch.full((num_nodes,), -100, dtype=torch.long)
    y[num_elec:] = torch.from_numpy(tm_label)
    tm_mask = torch.zeros(num_nodes, dtype=torch.bool)
    tm_mask[num_elec:] = True

    data = Data(
        x=x_scaled,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        tm_mask=tm_mask,
        num_elec=num_elec,
        pacing_info=sim["path"],
    )
    # elec_coords stays a top-level attribute (matches preprocess_graphs.py's
    # schema and is what the existing pipeline reads). The surface-only extras
    # live inside meta so PyG's DataLoader never tries to collate them: tm_coords
    # are the real per-node surface params the eval-time interp maps onto;
    # tm_xyz / elec_xyz are kept for 3D overlays.
    data["elec_coords"] = elec_tp.astype(np.float64)
    data["meta"] = {
        "surface": True,
        "rx": sim["rx"], "ry": sim["ry"], "rz": sim["rz"],
        "coord_scale": coord_scale,
        "num_tm": num_tm, "num_elec": num_elec,
        "n_fib": int(tm_label.sum()),
        "tm_coords": tm_tp.astype(np.float64),
        "tm_xyz": tm_xyz.astype(np.float64),
        "elec_xyz": sim["electrodes"].astype(np.float64),
    }
    return data


# ------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file-pattern", default="sim_results/*/phie_surface_*.npz",
                   help="Glob for curved-surface phie .npz files.")
    p.add_argument("--out-dir", default="surface_graphs")
    p.add_argument("--electrode-to-mesh-ratio", type=float, default=9.0,
                   help="Target #mesh-nodes per electrode after downsampling (default 9:1).")
    p.add_argument("--t-trunc-start", type=int, default=0)
    p.add_argument("--t-trunc-end", type=int, default=None)
    p.add_argument("--k-elec", type=int, default=20,
                   help="kNN electrodes per TM node for elec->TM edges (matches preprocess).")
    p.add_argument("--k-tm", type=int, default=8,
                   help="kNN neighbours per TM node for TM-TM edges (0 to disable).")
    p.add_argument("--no-tm-tm", action="store_true", help="Drop TM-TM edges.")
    p.add_argument("--w-max", type=float, default=500.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--coord-scale", type=float, default=1.0,
                   help="Multiplier on the edge coords; tune so edge_attr=1/dist "
                        "sits in the range the flat-trained model saw. The build "
                        "prints a suggested value to match the flat elec->TM median.")
    p.add_argument("--edge-space", default="xyz", choices=["thetaphi", "xyz"],
                   help="Coordinate space for kNN/inverse-distance edges. 'xyz' uses "
                        "real 3D chord distance (faithful surface proximity); "
                        "'thetaphi' uses the distorted (theta,phi) metric (default).")
    p.add_argument("--tm-tm-scale", type=float, default=1.0,
                   help="Extra multiplier on TM-TM edge weights only. coord_scale "
                        "preserves the elec->TM:TM-TM ratio; use this to match BOTH "
                        "flat targets and avoid over-smoothing. The build prints a "
                        "suggested value to match the flat TM-TM median (1.554).")
    p.add_argument("--scaling", default="std", choices=["std", "min-max"])
    p.add_argument("--t-max", type=int, default=1500)
    p.add_argument("--no-fib-bias", action="store_true",
                   help="Disable fibrotic-biased cluster representatives "
                        "(then small patches may vanish in downsampling).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(args.file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.file_pattern}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "graph_index.txt"

    saved = []
    used_names = set()
    for f in files:
        stem = Path(f).stem
        name = stem + ".pt"
        if name in used_names:
            name = f"{Path(f).parent.name}__{stem}.pt"
        used_names.add(name)
        graph_path = out_dir / name

        if args.skip_existing and graph_path.exists():
            print(f"Skip {f}: {graph_path.name} exists.")
            saved.append(graph_path)
            continue

        try:
            sim = load_surface_sim(f)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue

        data = build_graph(
            sim,
            electrode_to_mesh_ratio=args.electrode_to_mesh_ratio,
            t_trunc_start=args.t_trunc_start,
            t_trunc_end=args.t_trunc_end,
            k_elec=args.k_elec,
            k_tm=args.k_tm,
            w_max=args.w_max,
            eps=args.eps,
            coord_scale=args.coord_scale,
            scaling=args.scaling,
            T_max=args.t_max,
            fib_bias=not args.no_fib_bias,
            add_tm_tm=not args.no_tm_tm,
            seed=args.seed,
            edge_space=args.edge_space,
            tm_tm_scale=args.tm_tm_scale,
        )
        torch.save(data, graph_path)
        saved.append(graph_path)
        m = data["meta"]
        print(f"{f} -> {graph_path.name}: {m['num_elec']} elec + {m['num_tm']} TM "
              f"({m['n_fib']} fibrotic, {100 * m['n_fib'] / m['num_tm']:.1f}%), "
              f"{data.edge_index.shape[1]} edges, T={data.x.shape[1]}")

    with open(index_path, "w") as fh:
        for path_ in saved:
            fh.write(str(path_) + "\n")
    print(f"\nSaved {len(saved)} graph(s) to {out_dir} (index: {index_path}).")


if __name__ == "__main__":
    main()
