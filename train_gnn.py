import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import wandb

from torch.nn import Linear, CrossEntropyLoss
from torch_geometric.nn import GCNConv, GATv2Conv

from pathlib import Path
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os

target_shape = (30, 30)
t_trunc_start = 0
t_trunc_end = None
graph_type = 'dist-based-add-tm' #'dist-based'
delta = 3.0
eps = 1e-6
scaling = 'std'
third_label = True
task = 'classification'
noise = None
sparse_phie = None
T_max = 1500
loss_name = "ce"  # options: "ce", "dice", "dice_ce", "dice_bce", "focal"
dice_weight = 1.0
ce_weight = 1.0
bce_weight = 1.0
dice_smooth = 1e-5
focal_gamma = 2.0
focal_alpha = None
max_graphs = None  # Set to an int to limit how many graphs are used (e.g., 100)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN model")
    parser.add_argument(
        "--loss-name",
        default=loss_name,
        choices=["ce", "dice", "dice_ce", "dice_bce", "focal"],
        help="Loss function to use.",
    )
    parser.add_argument("--dice-weight", type=float, default=dice_weight)
    parser.add_argument("--ce-weight", type=float, default=ce_weight)
    parser.add_argument("--bce-weight", type=float, default=bce_weight)
    parser.add_argument("--dice-smooth", type=float, default=dice_smooth)
    parser.add_argument("--focal-gamma", type=float, default=focal_gamma)
    parser.add_argument("--focal-alpha", type=float, default=focal_alpha)
    parser.add_argument("--max-graphs", type=int, default=max_graphs)
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default=preprocess_dir,
        help="Directory containing graph_index.txt (default: preprocessed_graphs).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()

class LazyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, index_file, max_graphs=None, base_dir=None):
        with open(index_file, "r") as f:
            graph_paths = [line.strip() for line in f if line.strip()]
        if max_graphs is not None:
            graph_paths = graph_paths[:max_graphs]
        if base_dir is not None:
            base_dir = Path(base_dir)
            remapped_paths = []
            for p in graph_paths:
                p_path = Path(p)
                if p_path.is_absolute():
                    remapped_paths.append(p)
                    continue
                parts = p_path.parts
                if parts and parts[0] == base_dir.name:
                    p_path = Path(*parts[1:])
                candidate = (base_dir / p_path).resolve()
                if candidate.exists():
                    remapped_paths.append(str(candidate))
                    continue
                if parts and parts[0] == preprocess_dir:
                    candidate = (base_dir / Path(*parts[1:])).resolve()
                    remapped_paths.append(str(candidate))
                else:
                    remapped_paths.append(str(candidate))
            graph_paths = remapped_paths
        self.graph_paths = graph_paths

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx], weights_only=False)

preprocess_dir = "preprocessed_graphs"
def load_dataset(max_graphs, graphs_dir):
    index_file = Path(graphs_dir) / "graph_index.txt"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Missing {index_file}. Run the preprocessing script first to generate graphs."
        )
    dataset = LazyGraphDataset(index_file, max_graphs=max_graphs, base_dir=graphs_dir)
    print(f"Using lazy dataset with {len(dataset)} graphs.")
    return dataset

### Define model class based on model used by Rodrigo et al. 2022 for classifying EGMs
### https://www.sciencedirect.com/science/article/pii/S0010482522002438?via%3Dihub#appsec1 ##

class Rodrigo_GNN_edge_features(nn.Module):
    """
    1D CNN as specified.
    Uses LazyLinear so you don't need to precompute the flattened feature size.
    """
    def __init__(self, in_ch=1, gnn_type='GAT', out_ch=2, conv_ch=64, gnn_ch1=32, gnn_ch2=32, gnn_nlayer=2, 
                 edge_dim=1, kernel_size_1=64, kernel_size_2=32):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_ch, conv_ch, kernel_size_1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),
            # nn.BatchNorm1d(conv_ch),
            nn.GroupNorm(num_groups=4, num_channels=conv_ch),
            nn.Dropout(p=0.3),

            # Block 2
            nn.Conv1d(conv_ch, conv_ch, kernel_size_2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),
            # nn.BatchNorm1d(conv_ch),
            nn.GroupNorm(num_groups=8, num_channels=conv_ch),
            nn.Dropout(p=0.2),
        )

        # self.features = nn.Sequential(
        # # Block 1 (downsample by 2)
        # nn.Conv1d(in_ch, conv_ch, kernel_size=8, stride=2, padding=4),
        # nn.ReLU(inplace=True),
        # nn.BatchNorm1d(conv_ch),
        # # nn.GroupNorm(num_groups=4, num_channels=conv_ch),
        # nn.Dropout(p=0.3),

        # # Block 2 (downsample by 2 again => total /4)
        # nn.Conv1d(conv_ch, conv_ch, kernel_size=8, stride=2, padding=4),
        # nn.ReLU(inplace=True),
        # nn.BatchNorm1d(conv_ch),
        # # nn.GroupNorm(num_groups=8, num_channels=conv_ch),
        # nn.Dropout(p=0.2),
        # )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(conv_ch),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(conv_ch),
            nn.LayerNorm(conv_ch),
            nn.Dropout(p=0.3),
            # nn.Linear(conv_ch, gnn_ch),
            # nn.ReLU(inplace=True),
            # # nn.BatchNorm1d(gnn_ch),
            # nn.LayerNorm(gnn_ch),
            # nn.Dropout(p=0.2),  # no-op
        )
        self.gnn_type = gnn_type
        self.gcn = GCNConv(gnn_ch1, gnn_ch1)
        self.gat1 = GATv2Conv(conv_ch, gnn_ch1, edge_dim=1)
        self.gat2 = GATv2Conv(gnn_ch1, gnn_ch2, edge_dim=1)
        self.classifier = Linear(gnn_ch2, out_ch)
        self.gnn_nlayer=gnn_nlayer

    def forward(self, x, edge_index, edge_attr=None):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.head(x)

        if self.gnn_type == 'GAT':
            h = self.gat1(x, edge_index, edge_attr)#, return_attention_weights=True)
            h = h.relu()
            h = self.gat2(h, edge_index, edge_attr)#, return_attention_weights=True)
            h = h.relu()
            # Apply a final (linear) classifier.
            out = self.classifier(h)
            return out, h #, a

        elif self.gnn_type == 'GCN':
            for i in range(self.gnn_nlayer):
              h = self.gcn(x, edge_index)
              h = h.relu()
              x = h
            # Apply a final (linear) classifier.
            out = self.classifier(h)
            return out, h

class Rodrigo_GNN_edge_features2(nn.Module):
    """
    1D CNN as specified.
    Uses LazyLinear so you don't need to precompute the flattened feature size.
    """
    def __init__(
        self,
        in_ch=1,
        gnn_type='GAT',
        out_ch=2,
        conv_ch=16,
        gnn_ch1=16,
        gnn_ch2=16,
        gnn_nlayer=2,
        edge_dim=1,
        kernel_size_1=11,
        kernel_size_2=7,
        kernel_size_3=5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 (downsample)
            nn.Conv1d(in_ch, 32, kernel_size_1, stride=2, padding=kernel_size_1 // 2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.Dropout(p=0.2),

            # Block 2 (downsample)
            nn.Conv1d(32, 32, kernel_size_2, stride=2, padding=kernel_size_2 // 2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.Dropout(p=0.2),

            # Block 3 (downsample)
            nn.Conv1d(32, conv_ch, kernel_size_3, stride=2, padding=kernel_size_3 // 2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=4, num_channels=conv_ch),
            nn.Dropout(p=0.2),

            # Ensure width = 1
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),          # (N, 16)
            nn.LazyLinear(conv_ch),
            nn.ReLU(inplace=True),
            nn.LayerNorm(conv_ch),
            nn.Dropout(p=0.3),
        )
        self.gnn_type = gnn_type
        self.gcn = GCNConv(gnn_ch1, gnn_ch1)
        self.gat1 = GATv2Conv(conv_ch, gnn_ch1, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(gnn_ch1, gnn_ch2, edge_dim=edge_dim)
        self.classifier = Linear(gnn_ch2, out_ch)
        self.gnn_nlayer = gnn_nlayer

    def forward(self, x, edge_index, edge_attr=None):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.head(x)

        if self.gnn_type == 'GAT':
            h, a = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
            h = h.relu()
            h, a = self.gat2(h, edge_index, edge_attr, return_attention_weights=True)
            h = h.relu()
            out = self.classifier(h)
            return out, h, a

        elif self.gnn_type == 'GCN':
            for _ in range(self.gnn_nlayer):
                h = self.gcn(x, edge_index)
                h = h.relu()
                x = h
            out = self.classifier(h)
            return out, h
        
## train with multiple data files and test on some unseen file(s)
# train with clean (0 noise) and full electrodes
task='classification'
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _mask_logits_and_targets(logits, target, ignore_index):
    if ignore_index is None:
        return logits, target
    valid = target != ignore_index
    if valid.all():
        return logits, target
    return logits[valid], target[valid]

def _as_monai_batch(logits, target):
    if logits.dim() == 2:
        logits = logits.unsqueeze(0).transpose(1, 2)
        target = target.unsqueeze(0).unsqueeze(1)
    return logits, target

def build_loss_fn(
    name,
    num_classes,
    ignore_index=-100,
    dice_smooth=1e-5,
    dice_weight=1.0,
    ce_weight=1.0,
    bce_weight=1.0,
    focal_gamma=2.0,
    focal_alpha=None,
):
    if name == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index), None, None

    if name == "focal":
        if focal_alpha is not None:
            alpha = torch.tensor([1.0 - focal_alpha, focal_alpha], dtype=torch.float32)
        else:
            alpha = None

        def loss_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return logits.sum() * 0.0
            ce_loss = F.cross_entropy(
                logits,
                target,
                reduction="none",
                weight=alpha.to(logits.device) if alpha is not None else None,
            )
            pt = torch.exp(-ce_loss)
            focal = ((1.0 - pt) ** focal_gamma) * ce_loss
            return focal.mean()

        def component_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return (logits.sum() * 0.0,)
            ce_loss = F.cross_entropy(
                logits,
                target,
                reduction="none",
                weight=alpha.to(logits.device) if alpha is not None else None,
            )
            pt = torch.exp(-ce_loss)
            focal = ((1.0 - pt) ** focal_gamma) * ce_loss
            return (focal.mean(),)

        return loss_fn, component_fn, ["focal"]

    try:
        from monai.losses import DiceCELoss, DiceLoss
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for Dice-based losses. Install with `pip install monai`."
        ) from exc

    def _safe_zero(logits):
        return logits.sum() * 0.0

    if name == "dice":
        dice = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
            smooth_nr=dice_smooth,
            smooth_dr=dice_smooth,
        )

        def loss_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return _safe_zero(logits)
            logits, target = _as_monai_batch(logits, target)
            return dice(logits, target)

        def component_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return (_safe_zero(logits),)
            logits, target = _as_monai_batch(logits, target)
            return (dice(logits, target),)

        return loss_fn, component_fn, ["dice"]

    if name == "dice_ce":
        dice = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
            smooth_nr=dice_smooth,
            smooth_dr=dice_smooth,
        )
        ce = CrossEntropyLoss()
        dice_ce = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
            lambda_dice=dice_weight,
            lambda_ce=ce_weight,
            smooth_nr=dice_smooth,
            smooth_dr=dice_smooth,
        )

        def loss_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return _safe_zero(logits)
            logits, target = _as_monai_batch(logits, target)
            return dice_ce(logits, target)

        def component_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return (_safe_zero(logits), _safe_zero(logits))
            logits, target = _as_monai_batch(logits, target)
            ce_logits = logits.squeeze(0).transpose(0, 1)
            ce_target = target.squeeze(0).squeeze(0)
            return (dice(logits, target), ce(ce_logits, ce_target))

        return loss_fn, component_fn, ["dice", "ce"]

    if name == "dice_bce":
        dice = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
            smooth_nr=dice_smooth,
            smooth_dr=dice_smooth,
        )
        bce = nn.BCEWithLogitsLoss()

        def loss_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return _safe_zero(logits)
            logits, target = _as_monai_batch(logits, target)
            dice_loss = dice(logits, target)
            target_fg = (target == 1).float().squeeze(0).squeeze(0)
            bce_logits = logits[:, 1, :].squeeze(0)
            bce_loss = bce(bce_logits, target_fg)
            return dice_weight * dice_loss + bce_weight * bce_loss

        def component_fn(logits, target):
            logits, target = _mask_logits_and_targets(logits, target, ignore_index)
            if logits.numel() == 0:
                return (_safe_zero(logits), _safe_zero(logits))
            logits, target = _as_monai_batch(logits, target)
            dice_loss = dice(logits, target)
            target_fg = (target == 1).float().squeeze(0).squeeze(0)
            bce_logits = logits[:, 1, :].squeeze(0)
            bce_loss = bce(bce_logits, target_fg)
            return (dice_loss, bce_loss)

        return loss_fn, component_fn, ["dice", "bce"]

    raise ValueError(f"Unknown loss name: {name}")

def main(run, dataset, savefig, target_shape, seed, dim):
    if seed is None:
        seed = random.randrange(2**32)
    set_seed(seed)

    wandb_mode = os.getenv("WANDB_MODE")
    if wandb_mode is None:
        if os.getenv("WANDB_API_KEY"):
            wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=False)
            wandb_mode = "online"
        else:
            wandb_mode = "offline"

    wandb_run = wandb.init(
        project="gnn_cardiac_fibrosis",
        name=str(run),
        mode=wandb_mode,
        config={
            "seed": seed,
            "num_epochs": 1000,
            "batch_size": 4,
            "lr": 1e-4,
            "gnn_type": "GAT",
            "graph_type": graph_type,
            "delta": delta,
            "scaling": scaling,
            "target_shape": target_shape,
            "loss_name": loss_name,
            "dice_weight": dice_weight,
            "ce_weight": ce_weight,
            "bce_weight": bce_weight,
            "dice_smooth": dice_smooth,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "max_graphs": max_graphs,
        },
    )

    # out_dir = Path("benchmark_results") / f"0noise_100pct_elec_no_V_link" #f"anisotropy"
    out_dir = Path(run)
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_model_prediction_2d(model, graph, device, target_shape, epoch):
        # Visualise on 2D grid if you used target_shape
        model.to(device).eval()
        sample_graph = graph.clone()
        num_elec = sample_graph.num_elec
        with torch.no_grad():
            g = sample_graph.to(device)
            logits = model(g.x, g.edge_index, getattr(g, 'edge_attr', None))[0]
            preds = logits.argmax(dim=1)
        # Only TM nodes carry labels
        tm_mask = sample_graph.tm_mask
        y_true = sample_graph.y[tm_mask].cpu().numpy()
        y_pred = preds[tm_mask].cpu().numpy()
        if target_shape==None:
          target_shape=(100,100)
        plt.figure(figsize=(6,5))
        # Ground truth
        plt.contour(
            y_true.reshape(target_shape),
            levels=[0.5],
            colors='red',
            linewidths=1.5,
            origin='lower'
        )
        # Pred
        plt.imshow(y_pred.reshape(target_shape), cmap='gray', origin='lower')
        plt.title(f'Predicted Fibrotic Region with GT Contours - Epoch {epoch}')
        plt.colorbar(ticks=[0,1])
        plt.tight_layout()
        plt.show()
        plt.close()

    #%% Train/Validation split by files (80/20)
    n_total = len(dataset)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, n_total - n_train)
    print(n_train)
    print(n_val)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    num_workers = 3
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        exclude_keys=['pacing_info','meta'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=4,
        exclude_keys=['pacing_info','meta'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    #%% Model / Optimizer / Loss
    in_channels = dataset[0].x.shape[1]
    # model = ElectrodeTempConvGNN(
    #     in_channels=in_channels,
    #     hidden_channels=16,
    #     out_channels=2
    # ).to(device)
    # model = ElectrodeResNetGNN(
    #     in_channels=1,
    #     out_channels=2, # number of classes: fibrotic or not
    #     hidden_channels=16,
    #     resnet_cfg=dict(n_block=4, base_filters=32, kernel_size=5),
    # ).to(device)

    # model = ResNet_GNN(
    #     in_channels=1,
    #     out_channels=2, # number of classes: fibrotic or not
    #     hidden_channels=32,
    #     gnn_type = 'GAT', #'GAT'
    #     resnet_cfg=dict(n_block=3, base_filters=32, kernel_size=5),
    # ).to(device)

    model = Rodrigo_GNN_edge_features(
        in_ch=1,
        out_ch=2 if task == 'classification' else 1, # number of classes: fibrotic or not, or a regressed value
        # conv_ch=conv_ch,
        # gnn_ch1=gnn_ch1,
        # gnn_ch2=gnn_ch2,
        gnn_type='GAT', #'GCN'
        gnn_nlayer=2,
        edge_dim=1,
        # kernel_size_1=ks1,
        # kernel_size_2=ks2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn, component_fn, component_names = build_loss_fn(
        loss_name,
        num_classes=2,
        ignore_index=-100,
        dice_smooth=dice_smooth,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
        bce_weight=bce_weight,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )

    #%% Training loop
    num_epochs = 1000
    train_losses = []
    val_losses = []
    train_component_losses = {name: [] for name in (component_names or [])}
    val_component_losses = {name: [] for name in (component_names or [])}
    best_val = float('inf')
    best_state = None

    for epoch in range(1, num_epochs + 1):

        # ---- Train ----
        model.train()
        running = 0.0
        running_components = {name: 0.0 for name in (component_names or [])}
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None))
            # out: (sum_nodes_in_batch, 2)l
            loss = loss_fn(out[0], batch.y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if component_fn is not None:
                comps = component_fn(out[0], batch.y)
                for name, value in zip(component_names, comps):
                    running_components[name] += value.item()
        epoch_train_loss = running / len(train_loader)
        train_losses.append(epoch_train_loss)
        for name in running_components:
            train_component_losses[name].append(running_components[name] / len(train_loader))

        # ---- Val ----
        model.eval()
        with torch.no_grad():
            running = 0.0
            running_components = {name: 0.0 for name in (component_names or [])}
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None))
                loss = loss_fn(out[0], batch.y)
                running += loss.item()
                if component_fn is not None:
                    comps = component_fn(out[0], batch.y)
                    for name, value in zip(component_names, comps):
                        running_components[name] += value.item()
            epoch_val_loss = running / max(1, len(val_loader))
            val_losses.append(epoch_val_loss)
            for name in running_components:
                val_component_losses[name].append(running_components[name] / max(1, len(val_loader)))

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
            }
        )

        if epoch % 200 == 0:
            print(f"Epoch {epoch:04d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")
            g0 = batch.to_data_list()[0]
            #plot_model_prediction(model, g0, device, target_shape, epoch)
            # or plot all validation graphs:
            if dim==2:
              for i, g in enumerate(batch.to_data_list()):
                plot_model_prediction_2d(model, g, device, target_shape, epoch)

        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                  "epoch": epoch,
                  "val_loss": epoch_val_loss,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
              }, out_dir / f"best_model_{run}.pt")
            wandb.log({"best_val_loss": best_val})
        if epoch > 500 and epoch % 50 == 0:
            torch.save({
                  "epoch": num_epochs,
                  "val_loss": val_losses[-1] if len(val_losses) > 0 else None,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
              }, out_dir / f"final_model_{run}.pt")


    #%% Plot losses
    plt.figure()
    plt.semilogy(train_losses, label='Train Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.title(f'{loss_name} loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if savefig:
      plt.savefig(out_dir / f"losses_{run}.png")
    plt.show()
    plt.close()

    if component_names:
        plt.figure()
        for name in component_names:
            plt.semilogy(train_component_losses[name], label=f"Train {name}")
            plt.semilogy(val_component_losses[name], label=f"Validation {name}")
        plt.title(f"{loss_name} components over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(out_dir / f"losses_{run}_{loss_name}_components.png")
        plt.show()
        plt.close()

    #%% Evaluation on validation graph (qualitative & metrics)
    for i in range(len(val_set)):
        # model.load_state_dict(best_state) # use best or last model
        model.to(device).eval()
        sample_graph = val_set[i].clone()
        num_elec = sample_graph.num_elec
        with torch.no_grad():
            g = sample_graph.to(device)
            logits = model(g.x, g.edge_index, getattr(g, 'edge_attr', None))[0]
            preds = logits.argmax(dim=1)

        # Only TM nodes carry labels
        tm_mask = sample_graph.tm_mask
        y_true = sample_graph.y[tm_mask].cpu().numpy()
        y_pred = preds[tm_mask].cpu().numpy()

        report = classification_report(y_true, y_pred, labels=[0, 1],target_names=['non-fibrotic', 'fibrotic'])
        # Print to console
        print(report)
        # Save to a text file
        with open(out_dir / f"classification_report_{i}_{run}.txt", "w") as f:
            f.write(report)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False,
                    xticklabels=['non-fibrotic','fibrotic'],
                    yticklabels=['non-fibrotic','fibrotic'],
                    annot_kws={"size": 20})
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.title(f'Validation Confusion Matrix {i}', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if savefig:
          plt.savefig(out_dir / f"validation_confusion_matrix_{i}_{run}.png")
        plt.show()
        plt.close()

        # Visualise on 2D grid if you used target_shape
        if target_shape==None:
          target_shape = (100,100)
        if dim ==2:
            plt.figure(figsize=(6,5))
            # Pred
            elec_coords = sample_graph.elec_coords
            plt.imshow(y_pred.reshape(target_shape),
                       extent=[0, 10, 0, 10],
                       cmap='gray', origin='lower', aspect='equal')
            plt.colorbar(ticks=[0,1])
            x_phys = np.linspace(0, 10, target_shape[0])
            y_phys = np.linspace(0, 10, target_shape[1])
            Xg, Yg = np.meshgrid(x_phys, y_phys, indexing='xy')
            # Ground truth
            plt.contour(
                Xg, Yg,
                y_true.reshape(target_shape),
                levels=[0.5], colors='red', linewidths=1.5
            )
            plt.scatter(elec_coords[:, 0], elec_coords[:, 1],
                        c='red', label='Electrodes')
            plt.title(f'Predicted Fibrotic Region with GT contours - validation {i}')
            plt.tight_layout()
            if savefig:
                plt.savefig(out_dir / f"pred_vs_gt_{i}_{run}_w_elecs.png")
            plt.show()
            plt.close()
        elif dim==3:
            plot_model_prediction_3d_scatter(model, sample_graph, device, i)
    if wandb_run is not None:
        wandb_run.finish()
    return model

# for ks in [16,32,64]:
#     for gnn_ch in [16,32,64]:
#         for conv_ch in [16,32,64]:
#             dataset = torch.load(f"x_tm_zeros_noise_None_100pct_elec_with_V_link_delta_3dot0_2DAP_heter_D_x100.pt", weights_only=False)
#             model = main(f'x_tm_zeros_with_V_link_Conv_ks_{ks}_gnn_ch_{gnn_ch}_conv_ch_{conv_ch}',dataset,True,(30,30),seed=42,dim=2, \
#                  ks1=ks, ks2=ks, gnn_ch1=gnn_ch, gnn_ch2=gnn_ch)

if __name__ == "__main__":
    args = parse_args()
    loss_name = args.loss_name
    dice_weight = args.dice_weight
    ce_weight = args.ce_weight
    bce_weight = args.bce_weight
    dice_smooth = args.dice_smooth
    focal_gamma = args.focal_gamma
    focal_alpha = args.focal_alpha
    max_graphs = args.max_graphs

    dataset = load_dataset(max_graphs, args.graphs_dir)
    default_dir_name = Path(args.graphs_dir).name or Path(args.graphs_dir).as_posix()
    if args.run_name:
        run_name = f"{args.run_name}__{default_dir_name}"
    else:
        run_name = f"x_tm_zeros_with_V_link_{default_dir_name}_{len(dataset)}_{loss_name}"
    model = main(run_name, dataset, True, (30, 30), seed=args.seed, dim=2)