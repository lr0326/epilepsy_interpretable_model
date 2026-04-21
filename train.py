"""
Training script for the Interpretable Seizure Prediction Model
================================================================
Usage examples:

  # Quick smoke-test with synthetic data (no real EEG required):
  python train.py --synthetic --n_preictal 120 --n_interictal 240 --num_epochs 20

  # Train on real data stored in ./data/ (flat or class-based layout):
  python train.py --data_dir data/ --save_dir checkpoints/ --num_epochs 100

  # GPU training with custom hyperparameters:
  python train.py --synthetic --device cuda --batch_size 64 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from config import DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG
from dataset import EEGSequenceDataset, generate_synthetic_data, load_data_from_dir
from features import N_FEATURES, extract_features_batch
from loss import compute_loss
from model import InterpretableSeizurePredictor


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ─── Progress wrapper ─────────────────────────────────────────────────────────

def _iter(iterable, desc: str = ""):
    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


# ─── One epoch of training ────────────────────────────────────────────────────

def train_one_epoch(
    model: InterpretableSeizurePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kwargs: Dict,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        mean_loss: Average total loss over all batches.
        auc:       ROC-AUC score on the training set (0.5 if sklearn unavailable).
    """
    model.train()
    total_loss = 0.0
    all_probs: list = []
    all_labels: list = []

    for batch in _iter(loader, desc="Train"):
        x    = batch["eeg"].to(device)    # [B, N, C, T]
        feat = batch["feat"].to(device)   # [B, N, C, K]
        y    = batch["label"].to(device)  # [B, 1]

        outputs   = model(x, feat)
        loss_dict = compute_loss(outputs, y, **loss_kwargs)

        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss_dict["total_loss"].item()
        all_probs.append(outputs["risk_prob"].detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    probs  = np.concatenate(all_probs).flatten()
    labels = np.concatenate(all_labels).flatten()
    auc    = _safe_auc(labels, probs)

    return total_loss / len(loader), auc


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: InterpretableSeizurePredictor,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: Dict,
) -> Dict[str, float]:
    """Evaluate the model on *loader* and return a metrics dictionary."""
    model.eval()
    total_loss = 0.0
    all_probs: list = []
    all_labels: list = []

    for batch in _iter(loader, desc="Eval "):
        x    = batch["eeg"].to(device)
        feat = batch["feat"].to(device)
        y    = batch["label"].to(device)

        outputs   = model(x, feat)
        loss_dict = compute_loss(outputs, y, **loss_kwargs)
        total_loss += loss_dict["total_loss"].item()

        all_probs.append(outputs["risk_prob"].cpu().numpy())
        all_labels.append(y.cpu().numpy())

    probs  = np.concatenate(all_probs).flatten()
    labels = np.concatenate(all_labels).flatten()
    preds  = (probs >= 0.5).astype(int)

    metrics: Dict[str, float] = {"loss": total_loss / len(loader)}
    metrics["auc"] = _safe_auc(labels, probs)
    if SKLEARN_AVAILABLE:
        metrics["acc"] = float(accuracy_score(labels, preds))
        metrics["f1"]  = float(f1_score(labels, preds, zero_division=0))
    else:
        metrics["acc"] = float((preds == labels.astype(int)).mean())
        metrics["f1"]  = 0.0

    return metrics


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if not SKLEARN_AVAILABLE or len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.5


# ─── Prototype initialisation ─────────────────────────────────────────────────

def init_prototypes(
    model: InterpretableSeizurePredictor,
    loader: DataLoader,
    device: torch.device,
    n_prototypes: int,
) -> None:
    """
    Initialise the prototype memory bank via k-means on training embeddings.

    Prototype labels are assigned by the majority class of the samples
    assigned to each cluster.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("  [warn] scikit-learn not found – skipping prototype init.")
        return

    model.eval()
    embeddings_list: list = []
    labels_list: list = []

    with torch.no_grad():
        for batch in _iter(loader, desc="Proto"):
            x    = batch["eeg"].to(device)
            feat = batch["feat"].to(device)
            y    = batch["label"].numpy().flatten()
            out  = model(x, feat)
            embeddings_list.append(out["z_context"].cpu().numpy())
            labels_list.append(y)

    embeddings = np.concatenate(embeddings_list, axis=0)  # [S, D]
    labels     = np.concatenate(labels_list,     axis=0)  # [S]

    n_clusters = min(n_prototypes, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    centers = kmeans.cluster_centers_.astype(np.float32)  # [K, D]

    # Majority-vote label for each cluster
    cluster_labels = []
    for k in range(n_clusters):
        mask = kmeans.labels_ == k
        maj  = int(labels[mask].mean() >= 0.5) if mask.sum() > 0 else 0
        cluster_labels.append(maj)

    # Pad if needed
    if n_clusters < n_prototypes:
        pad = np.random.randn(n_prototypes - n_clusters, centers.shape[1]).astype(np.float32) * 0.01
        centers = np.concatenate([centers, pad], axis=0)
        cluster_labels += [0] * (n_prototypes - n_clusters)

    with torch.no_grad():
        model.prototype_head.prototypes.data.copy_(
            torch.from_numpy(centers)
        )
        model.prototype_head.update_prototype_labels(
            torch.tensor(cluster_labels, dtype=torch.long)
        )

    n_pre = sum(cluster_labels)
    print(
        f"  Prototypes initialised: {n_clusters} clusters "
        f"({n_pre} preictal, {n_clusters - n_pre} interictal)"
    )


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


# ─── Main training routine ────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    # ── Load / generate data ─────────────────────────────────────────────
    if args.synthetic:
        print("Generating synthetic data …")
        eeg, labels = generate_synthetic_data(
            n_preictal=args.n_preictal,
            n_interictal=args.n_interictal,
            n_channels=args.num_channels,
            n_windows=args.n_windows,
            window_size=args.window_size,
            fs=args.fs,
            seed=args.seed,
        )
        print(
            f"  {eeg.shape[0]} samples  "
            f"(preictal={int(labels.sum())}, interictal={int((labels == 0).sum())})"
        )
    else:
        print(f"Loading data from {args.data_dir} …")
        eeg, labels = load_data_from_dir(
            args.data_dir,
            window_size=args.window_size,
            n_windows=args.n_windows,
        )
        print(f"  EEG shape: {eeg.shape}")

    # ── Feature extraction ───────────────────────────────────────────────
    feat_cache = os.path.join(
        "/tmp" if args.synthetic else args.data_dir, "feat_cache.npy"
    )
    if os.path.exists(feat_cache) and not args.synthetic:
        print(f"Loading cached features from {feat_cache} …")
        features = np.load(feat_cache)
    else:
        print("Extracting handcrafted features …")
        features = extract_features_batch(
            eeg, fs=args.fs, compute_plv=args.use_plv, verbose=True
        )
        if not args.synthetic:
            np.save(feat_cache, features)
            print(f"  Features cached to {feat_cache}")

    print(f"  Features shape: {features.shape}")

    # ── Build model ──────────────────────────────────────────────────────
    model_cfg = {
        "num_channels":    eeg.shape[2],
        "raw_dim":         args.raw_dim,
        "ch_dim":          args.ch_dim,
        "feat_dim":        features.shape[3],
        "feat_hidden_dim": args.feat_hidden_dim,
        "concept_dim":     args.concept_dim,
        "d_model":         args.d_model,
        "num_prototypes":  args.num_prototypes,
    }
    model = InterpretableSeizurePredictor(**model_cfg).to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # ── Split datasets ───────────────────────────────────────────────────
    n_total = len(labels)
    n_val   = max(1, int(n_total * args.val_ratio))
    n_test  = max(1, int(n_total * args.test_ratio))
    n_train = n_total - n_val - n_test

    gen = torch.Generator().manual_seed(args.seed)
    idx_train, idx_val, idx_test = random_split(
        range(n_total), [n_train, n_val, n_test], generator=gen
    )
    idx_train = list(idx_train)
    idx_val   = list(idx_val)
    idx_test  = list(idx_test)

    # Augmentation only for training
    train_ds = EEGSequenceDataset(
        eeg[idx_train], features[idx_train], labels[idx_train], augment=True
    )
    val_ds = EEGSequenceDataset(
        eeg[idx_val],   features[idx_val],   labels[idx_val],   augment=False
    )
    test_ds = EEGSequenceDataset(
        eeg[idx_test],  features[idx_test],  labels[idx_test],  augment=False
    )
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    loss_kwargs = {
        "lambda_feat":  args.lambda_feat,
        "lambda_ch":    args.lambda_ch,
        "lambda_temp":  args.lambda_temp,
        "lambda_proto": args.lambda_proto,
    }

    # ── Training loop ────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_auc  = 0.0
    patience_ctr  = 0
    history: list = []

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Prototype initialisation after a warm-up epoch
        if epoch == args.init_proto_epoch:
            print(f"\nEpoch {epoch}: initialising prototypes …")
            init_prototypes(model, train_loader, device, args.num_prototypes)

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, device, loss_kwargs, args.grad_clip
        )
        val_metrics = evaluate(model, val_loader, device, loss_kwargs)
        scheduler.step(val_metrics["auc"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.num_epochs}  "
            f"[{elapsed:5.1f}s]  "
            f"train loss={train_loss:.4f} auc={train_auc:.3f}  |  "
            f"val loss={val_metrics['loss']:.4f} auc={val_metrics['auc']:.3f} "
            f"acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f}"
        )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc,
             **{f"val_{k}": v for k, v in val_metrics.items()}}
        )

        # Save best checkpoint
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_ctr = 0
            save_checkpoint(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config":       model_cfg,
                    "best_val_auc":       best_val_auc,
                },
                os.path.join(args.save_dir, "best_model.pt"),
            )
            print(f"  ✓ Best model saved  (val AUC = {best_val_auc:.4f})")
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Final evaluation on test set ─────────────────────────────────────
    print("\n── Test-set evaluation ──")
    ckpt = torch.load(
        os.path.join(args.save_dir, "best_model.pt"), map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, loss_kwargs)
    print(
        f"Test AUC={test_metrics['auc']:.4f}  "
        f"Acc={test_metrics['acc']:.4f}  "
        f"F1={test_metrics['f1']:.4f}"
    )

    # Save results JSON
    results = {
        "best_val_auc":  best_val_auc,
        "test_metrics":  test_metrics,
        "model_config":  model_cfg,
        "history":       history,
    }
    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return model, results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    mc = DEFAULT_MODEL_CONFIG
    tc = DEFAULT_TRAIN_CONFIG

    p = argparse.ArgumentParser(
        description="Train Interpretable Seizure Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data_dir",     default=tc["data_dir"],
                   help="Directory with EEG data (ignored if --synthetic)")
    g.add_argument("--save_dir",     default=tc["save_dir"])
    g.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic data for a quick smoke-test")
    g.add_argument("--n_preictal",   type=int, default=100,
                   help="Preictal samples to generate (synthetic mode)")
    g.add_argument("--n_interictal", type=int, default=200,
                   help="Interictal samples to generate (synthetic mode)")
    g.add_argument("--use_plv",      action="store_true",
                   help="Compute PLV connectivity features (slower)")

    # EEG parameters
    g = p.add_argument_group("EEG parameters")
    g.add_argument("--num_channels", type=int,   default=mc["num_channels"])
    g.add_argument("--window_size",  type=int,   default=mc["window_size"])
    g.add_argument("--n_windows",    type=int,   default=mc["n_windows"])
    g.add_argument("--fs",           type=float, default=mc["fs"])

    # Model architecture
    g = p.add_argument_group("model architecture")
    g.add_argument("--raw_dim",         type=int, default=mc["raw_dim"])
    g.add_argument("--ch_dim",          type=int, default=mc["ch_dim"])
    g.add_argument("--feat_hidden_dim", type=int, default=mc["feat_hidden_dim"])
    g.add_argument("--concept_dim",     type=int, default=mc["concept_dim"])
    g.add_argument("--d_model",         type=int, default=mc["d_model"])
    g.add_argument("--num_prototypes",  type=int, default=mc["num_prototypes"])

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--batch_size",       type=int,   default=tc["batch_size"])
    g.add_argument("--lr",               type=float, default=tc["learning_rate"])
    g.add_argument("--weight_decay",     type=float, default=tc["weight_decay"])
    g.add_argument("--num_epochs",       type=int,   default=tc["num_epochs"])
    g.add_argument("--patience",         type=int,   default=tc["patience"])
    g.add_argument("--grad_clip",        type=float, default=tc["grad_clip"])
    g.add_argument("--val_ratio",        type=float, default=tc["val_ratio"])
    g.add_argument("--test_ratio",       type=float, default=tc["test_ratio"])
    g.add_argument("--seed",             type=int,   default=tc["seed"])
    g.add_argument("--num_workers",      type=int,   default=tc["num_workers"])
    g.add_argument("--device",           default=tc["device"])
    g.add_argument("--init_proto_epoch", type=int,   default=tc["init_prototypes_epoch"])

    # Loss weights
    g = p.add_argument_group("loss weights")
    g.add_argument("--lambda_feat",  type=float, default=tc["lambda_feat"])
    g.add_argument("--lambda_ch",    type=float, default=tc["lambda_ch"])
    g.add_argument("--lambda_temp",  type=float, default=tc["lambda_temp"])
    g.add_argument("--lambda_proto", type=float, default=tc["lambda_proto"])

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args)
