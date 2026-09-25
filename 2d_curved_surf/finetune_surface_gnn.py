"""
finetune_surface_gnn.py

Few-shot fine-tune the flat-trained CNN+GAT on a handful of LABELLED curved-surface
graphs, to fix the boundary-morphology false positives (see the boundary diagnostic:
amplitude transfers fine, but the no-flux rim waveform shape is out-of-distribution
for the flat-interior-trained encoder).

Staged unfreezing (cheapest first; escalate only if the rim FPs persist):
  * head        : train ONLY the classifier on top of the frozen CNN + frozen GAT.
                  Works iff the frozen conv features already place boundary-shape
                  and fibrosis-shape in separable regions (the morphology diagnostic
                  says they differ -> worth trying first).
  * head+gat    : also unfreeze the GAT stack (CNN stays frozen).
  * full        : unfreeze everything incl. the CNN conv stack (freeze_cnn=False) --
                  let the features themselves adapt. Most capacity, most overfitting
                  risk on ~10 graphs; the last resort.

Protocol: train on the fine-tune graphs (a few held out for model selection), pick
the best epoch by VALIDATION average-precision, then report on the held-out test
graph -- both full-surface and interior-only (guard-band) so we can watch the rim
FPs drop without contaminating the headline number.

Usage:
    python finetune_surface_gnn.py \
        --gat-ckpt $EPHEMERAL/cnn_gat_runs/pretrained_elec_cnn_gat_lowerD_n6000_FIXED/best_model.pt \
        --ft-graphs surface_graphs_ft \
        --test-graphs surface_graphs_xyz_bal \
        --unfreeze head --epochs 100 --out-dir $EPHEMERAL/surface_finetune_head
"""

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# eval_surface_gnn imports the training repo (train_cnn_electrode); ensure it is
# importable even when PYTHONPATH is not carried into the process (e.g. background).
import sys as _sys
_sys.path.insert(0, "/Volumes/cc8418/home/GNN_cardiac_fibrosis")

# reuse the exact loader + metric helpers the evaluator uses, so train==eval
from eval_surface_gnn import (
    load_surface_model,
    best_f1_threshold,
    tm_edge_distance,
    plot_surface_overlay,
)


# --------------------------------------------------------------- data

def read_index(graphs_dir):
    idx = Path(graphs_dir) / "graph_index.txt"
    if not idx.exists():
        raise FileNotFoundError(f"Missing {idx}; run surface_to_graph.py first.")
    return [Path(l.strip()) for l in idx.read_text().splitlines() if l.strip()]


def resolve_paths(entries):
    """Each entry is either a graphs dir (-> read its graph_index.txt) or a single
    .pt graph. Lets --test-graphs mix several dirs / individual graphs."""
    paths = []
    for e in entries or []:
        pe = Path(e)
        if pe.suffix == ".pt":
            paths.append(pe)
        else:
            paths.extend(read_index(pe))
    return paths


def forward_graph(model, g, device):
    """One graph -> (tm_logits, tm_labels, edge_dist). Mirrors eval_surface_gnn."""
    logits, _ = model(
        g.x.to(device), g.edge_index.to(device),
        getattr(g, "edge_attr", None),
        tm_mask=g.tm_mask.to(device),
        elec_coords=g["elec_coords"],
        tm_coords=g["meta"]["tm_coords"],
    )
    tm = g.tm_mask.bool()
    labels = g.y[tm].to(device)
    edge_dist = tm_edge_distance(g["meta"])
    return logits[tm], labels, edge_dist


@torch.no_grad()
def plot_test_overlays(model, paths, device, thr, out_dir):
    """Multi-angle GT-vs-prediction overlay per test graph, prediction drawn at the
    fine-tuned model's operating threshold. The whole surface is shown (rim NOT
    grayed) so we can see directly whether the boundary false positives are gone."""
    model.eval()
    for gp in paths:
        g = torch.load(gp, map_location=device, weights_only=False)
        tm_logits, y, _ = forward_graph(model, g, device)
        probs = F.softmax(tm_logits, dim=1)[:, 1].cpu().numpy()
        preds_bin = (probs >= thr).astype(int)
        meta = g["meta"]
        plot_surface_overlay(
            np.asarray(meta["tm_xyz"], dtype=float),
            np.asarray(meta["elec_xyz"], dtype=float),
            preds_bin, y.cpu().numpy(),
            out_dir / f"overlay_{gp.stem}.png",
            thr_tag=f" @ thr {thr:.3f}")
    print(f"wrote {len(paths)} test overlay(s) to {out_dir}")


@torch.no_grad()
def evaluate(model, paths, device, guard_band=0.0):
    """Pooled probs/labels over a set of graphs -> AP / ROC-AUC, full + interior."""
    model.eval()
    probs, labels, edge = [], [], []
    for gp in paths:
        g = torch.load(gp, map_location=device, weights_only=False)
        tm_logits, y, ed = forward_graph(model, g, device)
        keep = (y != -100).cpu().numpy()
        probs.append(F.softmax(tm_logits, dim=1)[:, 1].cpu().numpy()[keep])
        labels.append(y.cpu().numpy()[keep])
        edge.append(ed[keep])
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    edge = np.concatenate(edge)

    def metrics(mask):
        yl, yp = labels[mask], probs[mask]
        if np.unique(yl).size < 2:
            return dict(ap=float("nan"), auc=float("nan"), n=int(mask.sum()))
        return dict(ap=average_precision_score(yl, yp),
                    auc=roc_auc_score(yl, yp), n=int(mask.sum()))

    full = metrics(np.ones_like(labels, bool))
    interior = metrics(edge >= guard_band) if guard_band > 0 else full
    return full, interior, probs, labels, edge


# --------------------------------------------------------------- freezing

def set_trainable(model, mode):
    """Stage the unfreezing. Returns the list of trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.freeze_cnn = True  # keep CNN under no_grad in _encode_electrodes

    # the trainable classifier ('new' head) or the reused CNN classifier ('cnn').
    if model.classifier is not None:
        for p in model.classifier.parameters():
            p.requires_grad_(True)
    elif getattr(model.cnn, "classifier", None) is not None:
        for p in model.cnn.classifier.parameters():
            p.requires_grad_(True)

    if mode in ("head+gat", "full"):
        for p in model.gats.parameters():
            p.requires_grad_(True)
    if mode == "full":
        model.freeze_cnn = False           # let CNN forward track gradients
        for p in model.cnn.parameters():
            p.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


def class_weights(paths, device, override=None):
    """Inverse-frequency 2-class weights from the training labels (fibrosis is the
    minority), unless --pos-weight overrides the fibrotic-class weight."""
    n0 = n1 = 0
    for gp in paths:
        g = torch.load(gp, map_location="cpu", weights_only=False)
        y = g.y[g.tm_mask.bool()].numpy()
        y = y[y != -100]
        n1 += int((y == 1).sum()); n0 += int((y == 0).sum())
    if override is not None:
        w = torch.tensor([1.0, float(override)], device=device)
    else:
        tot = n0 + n1
        w = torch.tensor([tot / (2 * max(n0, 1)), tot / (2 * max(n1, 1))],
                         device=device)
    print(f"train class balance: healthy={n0} fibrotic={n1} "
          f"({100 * n1 / max(n0 + n1, 1):.1f}%) -> loss weights {w.tolist()}")
    return w


# --------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gat-ckpt", required=True, help="flat-trained best_model.pt.")
    p.add_argument("--ft-graphs", required=True, help="dir of curved fine-tune graphs.")
    p.add_argument("--test-graphs", nargs="+", default=None,
                   help="held-out curved test graph(s): one or more dirs (each read "
                        "via graph_index.txt) and/or individual .pt files.")
    p.add_argument("--out-dir", default="surface_finetune")
    p.add_argument("--unfreeze", default="head",
                   choices=["head", "head+gat", "full"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-val", type=int, default=2,
                   help="how many fine-tune graphs to hold out for model selection.")
    p.add_argument("--pos-weight", type=float, default=None,
                   help="override the fibrotic-class loss weight (else inverse-freq).")
    p.add_argument("--guard-band", type=float, default=0.15,
                   help="interior-only reporting threshold (rim distance), to track "
                        "the boundary FPs separately.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model (reuse the evaluator's loader so weights/arch match exactly) ----
    model, mcfg = load_surface_model(args.gat_ckpt, device)
    print(f"loaded: tm_feature_mode={mcfg.get('tm_feature_mode')} "
          f"num_gat_layers={mcfg.get('num_gat_layers')} "
          f"classifier_head={mcfg.get('classifier_head')}")
    trainable = set_trainable(model, args.unfreeze)
    n_train_p = sum(p.numel() for p in trainable)
    print(f"unfreeze='{args.unfreeze}' -> {n_train_p} trainable params "
          f"({len(trainable)} tensors)")

    # ---- data split ----
    ft_paths = read_index(args.ft_graphs)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ft_paths))
    n_val = min(args.n_val, max(0, len(ft_paths) - 1))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_paths = [ft_paths[i] for i in train_idx]
    val_paths = [ft_paths[i] for i in val_idx]
    test_paths = resolve_paths(args.test_graphs)
    print(f"{len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test graph(s)")

    weights = class_weights(train_paths, device, override=args.pos_weight)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # ---- train ----
    history = {"epoch": [], "train_loss": [], "val_ap": [],
               "test_ap": [], "test_ap_interior": []}
    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_idx)  # reshuffle order each epoch
        epoch_loss = 0.0
        for i in train_idx:
            g = torch.load(ft_paths[i], map_location=device, weights_only=False)
            tm_logits, y, _ = forward_graph(model, g, device)
            loss = criterion(tm_logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss)
        epoch_loss /= max(len(train_idx), 1)

        val_full, _, *_ = (evaluate(model, val_paths, device)
                           if val_paths else ({"ap": float("nan")},) * 5)
        val_ap = val_full["ap"]
        rec = f"epoch {epoch:03d} | loss {epoch_loss:.4f} | val AP {val_ap:.4f}"

        test_ap = test_ap_int = float("nan")
        if test_paths:
            t_full, t_int, *_ = evaluate(model, test_paths, device,
                                         guard_band=args.guard_band)
            test_ap, test_ap_int = t_full["ap"], t_int["ap"]
            rec += (f" | test AP {test_ap:.4f} "
                    f"(interior {test_ap_int:.4f}, n={t_int['n']})")
        print(rec)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_ap"].append(val_ap)
        history["test_ap"].append(test_ap)
        history["test_ap_interior"].append(test_ap_int)

        score = val_ap if val_paths else -epoch_loss
        if np.isfinite(score) and score > best_val:
            best_val, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())

    # ---- save best + final, report ----
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nbest epoch {best_epoch} (val AP {best_val:.4f})")
    if test_paths:
        t_full, t_int, probs, labels, edge = evaluate(
            model, test_paths, device, guard_band=args.guard_band)
        thr, pr, re, f1 = best_f1_threshold(labels, probs)
        print(f"TEST  full: AP {t_full['ap']:.4f} AUC {t_full['auc']:.4f} | "
              f"interior(gb{args.guard_band}): AP {t_int['ap']:.4f} AUC {t_int['auc']:.4f}")
        print(f"TEST  max-F1 thr {thr:.4f}: precision {pr:.3f} recall {re:.3f} F1 {f1:.3f}")
        if len(test_paths) > 1:
            print("  per-graph:")
            for gp in test_paths:
                gf, gi, *_ = evaluate(model, [gp], device, guard_band=args.guard_band)
                print(f"    {gp.stem}: full AP {gf['ap']:.4f} AUC {gf['auc']:.4f} | "
                      f"interior AP {gi['ap']:.4f} (n={gi['n']})")
        plot_test_overlays(model, test_paths, device, thr, out_dir)

    ckpt = torch.load(args.gat_ckpt, map_location="cpu", weights_only=False)
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["finetune"] = {"unfreeze": args.unfreeze, "epochs": args.epochs,
                        "lr": args.lr, "best_epoch": best_epoch,
                        "n_train": len(train_paths)}
    torch.save(ckpt, out_dir / "best_model.pt")
    print(f"saved fine-tuned model -> {out_dir / 'best_model.pt'} "
          f"(eval with eval_surface_gnn.py)")

    # ---- training curves ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["val_ap"], label="val AP")
    if test_paths:
        ax.plot(history["epoch"], history["test_ap"], label="test AP (full)")
        ax.plot(history["epoch"], history["test_ap_interior"],
                label="test AP (interior)")
    ax.axvline(best_epoch, color="k", ls=":", lw=1, label=f"best ({best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("average precision")
    ax.set_title(f"fine-tune ({args.unfreeze})"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "finetune_curves.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_dir / 'finetune_curves.png'}")


if __name__ == "__main__":
    main()
