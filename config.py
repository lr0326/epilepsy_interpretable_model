"""Default configuration for the Epilepsy Interpretable Model."""

# ── Model architecture ────────────────────────────────────────────────────────
DEFAULT_MODEL_CONFIG = {
    # EEG input parameters
    "num_channels": 19,           # C: number of EEG channels
    "window_size": 256,           # T: samples per window  (1 s @ 256 Hz)
    "n_windows": 10,              # N: consecutive windows per sample
    "fs": 256.0,                  # sampling frequency (Hz)
    # Handcrafted feature parameters
    "n_handcrafted_features": 15, # K: features per channel (see features.py)
    # Model dimensions
    "raw_dim": 128,               # D1: raw encoder output dim
    "ch_dim": 64,                 # D_ch: per-channel embedding dim
    "feat_hidden_dim": 128,       # D2: feature encoder hidden dim
    "concept_dim": 32,            # Kc: concept embedding dim
    "d_model": 256,               # D: fused representation dim
    "num_prototypes": 50,         # P: number of prototype embeddings
}

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_TRAIN_CONFIG = {
    # Optimisation
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "patience": 15,           # early-stopping patience (epochs without val improvement)
    "grad_clip": 1.0,         # gradient-clipping max norm
    # Loss weights
    "lambda_feat": 1e-3,
    "lambda_ch": 1e-3,
    "lambda_temp": 1e-3,
    "lambda_proto": 1e-3,
    # Data split
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    # Misc
    "seed": 42,
    "num_workers": 4,
    "device": "auto",         # "auto" | "cuda" | "cpu"
    "data_dir": "data",
    "save_dir": "checkpoints",
    # Prototype initialisation epoch (0 = disabled, 1 = after first epoch, …)
    "init_prototypes_epoch": 2,
}

# ── Inference / alarm ─────────────────────────────────────────────────────────
DEFAULT_ALARM_CONFIG = {
    "ewma_beta": 0.8,          # EWMA smoothing factor
    "alarm_threshold": 0.5,    # risk score threshold
    "min_consecutive": 3,      # consecutive windows above threshold to fire
    "refractory_period": 10,   # minimum windows between successive alarms
}
