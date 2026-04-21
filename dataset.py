"""
EEG Dataset utilities
======================
Provides:
  ``EEGSequenceDataset``   – PyTorch Dataset for (eeg, features, label) triples.
  ``create_window_sequences`` – Slice long EEG recordings into overlapping
                                 sequences of N consecutive windows.
  ``load_data_from_dir``      – Load pre-segmented numpy files from disk.
  ``generate_synthetic_data`` – Create synthetic EEG-like data for testing.

Data format expected by the model
-----------------------------------
  eeg:      float32  [S, N, C, T]
  features: float32  [S, N, C, K]
  labels:   int64    [S]           (0 = interictal,  1 = preictal)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Dataset class ────────────────────────────────────────────────────────────

class EEGSequenceDataset(Dataset):
    """
    Dataset that wraps pre-processed EEG window sequences.

    Args:
        eeg:      Raw EEG sequences            [S, N, C, T] float32
        features: Handcrafted feature arrays   [S, N, C, K] float32
        labels:   Binary seizure-risk labels   [S]          int64
        augment:  Whether to apply on-the-fly data augmentation.
    """

    def __init__(
        self,
        eeg: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        assert eeg.shape[0] == features.shape[0] == len(labels), (
            "eeg, features and labels must have the same number of samples"
        )
        self.eeg      = torch.from_numpy(eeg.astype(np.float32))
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels   = torch.from_numpy(labels.astype(np.int64))
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eeg  = self.eeg[idx]       # [N, C, T]
        feat = self.features[idx]  # [N, C, K]
        lbl  = self.labels[idx]    # scalar

        if self.augment:
            eeg = self._augment(eeg)

        return {
            "eeg":   eeg,
            "feat":  feat,
            "label": lbl.float().unsqueeze(0),  # [1] for BCE loss
        }

    @staticmethod
    def _augment(eeg: torch.Tensor) -> torch.Tensor:
        """Light augmentation: additive Gaussian noise + random amplitude scaling."""
        if torch.rand(1).item() < 0.4:
            eeg = eeg + torch.randn_like(eeg) * (0.01 * eeg.std())
        if torch.rand(1).item() < 0.4:
            eeg = eeg * (0.9 + 0.2 * torch.rand(1).item())
        return eeg

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for a weighted sampler / loss."""
        n_pos  = int((self.labels == 1).sum())
        n_neg  = int((self.labels == 0).sum())
        total  = len(self.labels)
        w_pos  = total / (2 * n_pos)  if n_pos > 0 else 1.0
        w_neg  = total / (2 * n_neg)  if n_neg > 0 else 1.0
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


# ─── Windowing helper ─────────────────────────────────────────────────────────

def create_window_sequences(
    eeg_segments: List[np.ndarray],
    labels: List[int],
    window_size: int = 256,
    n_windows: int = 10,
    stride: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide over each EEG recording and cut out sequences of N consecutive windows.

    Args:
        eeg_segments: List of [C, T_total] arrays.
        labels:       Integer label (0/1) for each segment.
        window_size:  Samples per window (T).
        n_windows:    Windows per sequence (N).
        stride:       Stride in *samples* between the start of consecutive
                      sequences (can be smaller than window_size for overlap).

    Returns:
        eeg_seqs:   [S, N, C, T]
        seq_labels: [S]
    """
    sequences: List[np.ndarray] = []
    seq_labels: List[int] = []
    seq_len = window_size * n_windows

    for seg, lbl in zip(eeg_segments, labels):
        C, T_total = seg.shape
        for start in range(0, T_total - seq_len + 1, stride):
            chunk = seg[:, start : start + seq_len]   # [C, seq_len]
            # Reshape to [N, C, T]
            windows = chunk.reshape(C, n_windows, window_size)
            windows = windows.transpose(1, 0, 2)       # [N, C, T]
            sequences.append(windows)
            seq_labels.append(lbl)

    if not sequences:
        return np.empty((0, n_windows, 1, window_size), dtype=np.float32), np.empty(0, dtype=np.int64)

    return (
        np.stack(sequences, axis=0).astype(np.float32),
        np.array(seq_labels, dtype=np.int64),
    )


# ─── Data-loading utilities ───────────────────────────────────────────────────

def load_data_from_dir(
    data_dir: str,
    window_size: int = 256,
    n_windows: int = 10,
    stride: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data from *data_dir*.

    Accepted directory layouts
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Flat (pre-segmented)**::

        data_dir/
            X.npy   # [S, N, C, T]
            y.npy   # [S]

    **Class-based**::

        data_dir/
            preictal/    *.npy   # each file: [C, T_total]
            interictal/  *.npy

    Returns:
        eeg:    [S, N, C, T]  float32
        labels: [S]           int64
    """
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading pre-segmented data from {data_dir} …")
        return np.load(x_path).astype(np.float32), np.load(y_path).astype(np.int64)

    preictal_dir   = os.path.join(data_dir, "preictal")
    interictal_dir = os.path.join(data_dir, "interictal")
    if not (os.path.exists(preictal_dir) and os.path.exists(interictal_dir)):
        raise FileNotFoundError(
            f"Expected 'X.npy'/'y.npy' or 'preictal'/'interictal' sub-dirs in {data_dir}"
        )

    segments: List[np.ndarray] = []
    labels_raw: List[int] = []

    for fname in sorted(os.listdir(preictal_dir)):
        if fname.endswith(".npy"):
            segments.append(np.load(os.path.join(preictal_dir, fname)))
            labels_raw.append(1)
    for fname in sorted(os.listdir(interictal_dir)):
        if fname.endswith(".npy"):
            segments.append(np.load(os.path.join(interictal_dir, fname)))
            labels_raw.append(0)

    print(
        f"Loaded {labels_raw.count(1)} preictal and "
        f"{labels_raw.count(0)} interictal segments."
    )
    return create_window_sequences(segments, labels_raw, window_size, n_windows, stride)


# ─── Synthetic data generator (for smoke-tests and quick experiments) ─────────

def generate_synthetic_data(
    n_preictal: int = 100,
    n_interictal: int = 200,
    n_channels: int = 19,
    n_windows: int = 10,
    window_size: int = 256,
    fs: float = 256.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG-like sequences for testing.

    Pre-ictal samples are simulated with progressively increasing gamma-band
    activity and amplitude toward the end of the sequence.  Inter-ictal samples
    contain only background EEG noise.

    Returns:
        eeg:    [S, N, C, T]  float32  (shuffled)
        labels: [S]           int64
    """
    rng = np.random.RandomState(seed)
    t   = np.linspace(0, window_size / fs, window_size, endpoint=False)

    def _make_window(c_idx: int, win_idx: int, is_preictal: bool) -> np.ndarray:
        sig = (
            0.5 * np.sin(2 * np.pi * 2.0  * t)   # delta
            + 0.3 * np.sin(2 * np.pi * 6.0  * t)  # theta
            + 0.4 * np.sin(2 * np.pi * 10.0 * t)  # alpha
            + 0.2 * np.sin(2 * np.pi * 20.0 * t)  # beta
            + rng.normal(0, 0.10, window_size)     # broadband noise
        )
        if is_preictal:
            progress = (win_idx + 1) / n_windows
            sig += progress * (
                0.4 * np.sin(2 * np.pi * 45.0 * t)          # gamma burst
                + rng.normal(0, 0.05 * progress, window_size)
            )
        sig += rng.normal(0, 0.02, window_size)              # channel noise
        return sig.astype(np.float32)

    all_eeg:    List[np.ndarray] = []
    all_labels: List[int] = []

    for is_pre, n_samp in [(True, n_preictal), (False, n_interictal)]:
        lbl = 1 if is_pre else 0
        for _ in range(n_samp):
            seq = np.stack(
                [
                    np.stack(
                        [_make_window(c, n, is_pre) for c in range(n_channels)],
                        axis=0,
                    )
                    for n in range(n_windows)
                ],
                axis=0,
            )  # [N, C, T]
            all_eeg.append(seq)
            all_labels.append(lbl)

    # Shuffle
    idx = rng.permutation(len(all_eeg))
    eeg    = np.stack([all_eeg[i]    for i in idx], axis=0)
    labels = np.array([all_labels[i] for i in idx], dtype=np.int64)

    return eeg, labels
