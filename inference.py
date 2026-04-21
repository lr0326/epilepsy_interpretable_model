"""
Inference utilities for interpretable seizure prediction
=========================================================
Provides:
  ``predict``               – run the model on a single sample and return risk + explanations.
  ``build_explanation_report`` – convert raw model outputs into a human-readable dict.
  ``ewma_aggregate``        – smooth a sequence of per-window risk scores.
  ``alarm_logic``           – convert smoothed risks into binary alarms with refractory period.
  ``run_continuous_inference`` – end-to-end streaming inference over a long EEG recording.

Quick start
-----------
  >>> from model import InterpretableSeizurePredictor
  >>> from inference import predict, build_explanation_report
  >>> model = InterpretableSeizurePredictor(...)
  >>> ckpt  = torch.load("checkpoints/best_model.pt")
  >>> model.load_state_dict(ckpt["model_state_dict"])
  >>> outputs = predict(model, eeg_window_seq, feat_window_seq, device="cpu")
  >>> report  = build_explanation_report(outputs)
  >>> print(report)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from features import FEATURE_NAMES, N_FEATURES, extract_handcrafted_features
from model import InterpretableSeizurePredictor


# ─── Single-sample prediction ─────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: InterpretableSeizurePredictor,
    eeg: np.ndarray,
    features: Optional[np.ndarray] = None,
    fs: float = 256.0,
    device: str = "cpu",
    topk_proto: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Run a forward pass for a single EEG window sequence.

    Args:
        model:       Trained ``InterpretableSeizurePredictor``.
        eeg:         Raw EEG  –  shape [N, C, T]  (numpy float32).
        features:    Pre-computed handcrafted features  [N, C, K].
                     Computed on-the-fly if ``None``.
        fs:          Sampling frequency (used only when features is None).
        device:      Torch device string.
        topk_proto:  Number of top-k prototypes to retrieve.

    Returns:
        Dict of model output tensors (risk_prob, importances, prototypes …).
    """
    dev = torch.device(device)
    model.to(dev).eval()

    # Auto-compute handcrafted features if not provided
    if features is None:
        N, C, T = eeg.shape
        features = np.stack(
            [extract_handcrafted_features(eeg[n], fs=fs) for n in range(N)],
            axis=0,
        )  # [N, C, K]

    # Add batch dimension
    x_t    = torch.from_numpy(eeg[None].astype(np.float32)).to(dev)       # [1, N, C, T]
    feat_t = torch.from_numpy(features[None].astype(np.float32)).to(dev)  # [1, N, C, K]

    return model(x_t, feat_t, topk_proto=topk_proto)


# ─── Explanation report ───────────────────────────────────────────────────────

def build_explanation_report(
    outputs: Dict[str, torch.Tensor],
    feature_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    prototype_meta: Optional[List[Dict]] = None,
    topk: int = 3,
) -> Dict:
    """
    Convert raw model outputs (for a single sample, B=1) into a structured,
    human-readable explanation report.

    Args:
        outputs:        Dict returned by the model's ``forward``.
        feature_names:  Names of the Kc concept features.
                        Defaults to ``FEATURE_NAMES`` from features.py.
        channel_names:  EEG channel labels (e.g. standard 10-20 names).
                        Auto-generated ("CH1", "CH2", …) when ``None``.
        prototype_meta: List of metadata dicts, one per prototype
                        (e.g. ``{"patient": "chb01", "seizure_id": 3, "label": "preictal"}``).
        topk:           Number of top items to include in each category.

    Returns:
        report: Dict with keys
            ``risk_prob``, ``top_features``, ``top_channels``,
            ``top_time_windows``, ``top_prototypes``.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    # Extract for the first (and only) batch item  ─────────────────────
    feat_imp  = outputs["feat_importance"][0].mean(dim=0).detach().cpu()   # [Kc]
    ch_imp    = outputs["channel_importance"][0].mean(dim=0).detach().cpu()# [C]
    temp_imp  = outputs["temporal_importance"][0].detach().cpu()           # [N]
    proto_idx = outputs["prototype_topk_idx"][0].detach().cpu()            # [k]
    proto_val = outputs["prototype_topk_val"][0].detach().cpu()            # [k]
    risk_prob = float(outputs["risk_prob"][0].item())

    # Auto channel names ───────────────────────────────────────────────
    if channel_names is None:
        channel_names = [f"CH{i + 1}" for i in range(ch_imp.shape[0])]

    # Ranked indices ───────────────────────────────────────────────────
    feat_rank = torch.argsort(feat_imp,  descending=True)
    ch_rank   = torch.argsort(ch_imp,    descending=True)
    time_rank = torch.argsort(temp_imp,  descending=True)

    # Build report ─────────────────────────────────────────────────────
    top_features = [
        {"name":       feature_names[int(i)] if int(i) < len(feature_names) else f"concept_{i}",
         "importance": float(feat_imp[i])}
        for i in feat_rank[:topk]
    ]

    top_channels = [
        {"name":       channel_names[int(i)] if int(i) < len(channel_names) else f"CH{i}",
         "importance": float(ch_imp[i])}
        for i in ch_rank[:topk]
    ]

    top_time_windows = [
        {"window_index": int(i), "importance": float(temp_imp[i])}
        for i in time_rank[:topk]
    ]

    top_prototypes = []
    for idx, val in zip(proto_idx.tolist(), proto_val.tolist()):
        entry: Dict = {"prototype_id": int(idx), "similarity": float(val)}
        if prototype_meta is not None and int(idx) < len(prototype_meta):
            entry["meta"] = prototype_meta[int(idx)]
        top_prototypes.append(entry)

    return {
        "risk_prob":        risk_prob,
        "top_features":     top_features,
        "top_channels":     top_channels,
        "top_time_windows": top_time_windows,
        "top_prototypes":   top_prototypes,
    }


def print_explanation_report(report: Dict) -> None:
    """Pretty-print an explanation report to stdout."""
    bar = "─" * 52
    print(bar)
    print(f"  Seizure Risk Probability:  {report['risk_prob']:.4f}")
    print(bar)

    print("\n  Top concept features:")
    for item in report["top_features"]:
        print(f"    {item['name']:<24s}  {item['importance']:.4f}")

    print("\n  Top EEG channels:")
    for item in report["top_channels"]:
        print(f"    {item['name']:<24s}  {item['importance']:.4f}")

    print("\n  Most informative time windows:")
    for item in report["top_time_windows"]:
        print(f"    Window {item['window_index']:<3d}               {item['importance']:.4f}")

    print("\n  Most similar prototypes:")
    for item in report["top_prototypes"]:
        meta_str = ""
        if "meta" in item:
            meta_str = "  " + str(item["meta"])
        print(f"    Prototype {item['prototype_id']:<4d}  sim={item['similarity']:.4f}{meta_str}")
    print(bar)


# ─── Risk smoothing & alarm logic ─────────────────────────────────────────────

def ewma_aggregate(
    risk_seq: List[float],
    beta: float = 0.8,
) -> List[float]:
    """
    Exponentially-Weighted Moving Average smoothing of per-window risk scores.

    R_t = β · R_{t-1} + (1 − β) · r_t

    Args:
        risk_seq: Raw per-window risk scores in temporal order.
        beta:     Smoothing factor (higher ⇒ more smoothing).

    Returns:
        Smoothed risk sequence of the same length.
    """
    agg  = []
    prev = 0.0
    for r in risk_seq:
        prev = beta * prev + (1.0 - beta) * float(r)
        agg.append(prev)
    return agg


def alarm_logic(
    risk_seq: List[float],
    threshold: float = 0.5,
    min_consecutive: int = 3,
    refractory_period: int = 10,
) -> List[int]:
    """
    Convert a risk sequence into a binary alarm sequence.

    An alarm fires when:
      1. The risk score has been ≥ *threshold* for at least *min_consecutive*
         consecutive steps.
      2. No alarm has fired in the last *refractory_period* steps.

    Args:
        risk_seq:          Per-window risk scores (possibly EWMA-smoothed).
        threshold:         Risk threshold.
        min_consecutive:   Minimum consecutive high-risk windows to trigger.
        refractory_period: Minimum steps between successive alarms.

    Returns:
        List of binary alarm flags (1 = alarm fired, 0 = no alarm).
    """
    alarms      = [0] * len(risk_seq)
    consecutive = 0
    last_alarm  = -refractory_period

    for t, r in enumerate(risk_seq):
        if r >= threshold:
            consecutive += 1
        else:
            consecutive = 0

        if (
            consecutive >= min_consecutive
            and (t - last_alarm) >= refractory_period
        ):
            alarms[t]   = 1
            last_alarm  = t
            consecutive = 0   # reset counter after alarm

    return alarms


# ─── Continuous streaming inference ───────────────────────────────────────────

def run_continuous_inference(
    model: InterpretableSeizurePredictor,
    eeg_recording: np.ndarray,
    window_size: int = 256,
    n_windows: int = 10,
    stride_windows: int = 1,
    fs: float = 256.0,
    device: str = "cpu",
    ewma_beta: float = 0.8,
    alarm_threshold: float = 0.5,
    min_consecutive: int = 3,
    refractory_period: int = 10,
    topk_proto: int = 3,
) -> Dict:
    """
    Slide a sequence window over a long EEG recording and emit risk scores
    and alarms at each step.

    Args:
        model:             Trained model.
        eeg_recording:     Continuous EEG  [C, T_total].
        window_size:       Samples per single window (T).
        n_windows:         Number of windows per model input (N).
        stride_windows:    How many windows to advance between predictions
                           (1 = maximum temporal resolution).
        fs:                Sampling frequency.
        device:            Torch device.
        ewma_beta:         EWMA smoothing factor.
        alarm_threshold:   Risk threshold for alarm.
        min_consecutive:   Min consecutive steps above threshold to fire.
        refractory_period: Min steps between alarms.
        topk_proto:        Top-k prototypes to retrieve.

    Returns:
        Dict with keys:
            ``raw_risks``     – per-step raw risk scores
            ``smooth_risks``  – EWMA-smoothed risks
            ``alarms``        – binary alarm flags
            ``reports``       – list of explanation reports (one per step)
            ``step_times``    – start time (in samples) of each step
    """
    C, T_total   = eeg_recording.shape
    seq_len      = window_size * n_windows
    stride_samp  = window_size * stride_windows

    raw_risks: List[float]  = []
    reports:   List[Dict]   = []
    step_times: List[int]   = []

    for start in range(0, T_total - seq_len + 1, stride_samp):
        chunk    = eeg_recording[:, start : start + seq_len]    # [C, seq_len]
        # Reshape to [N, C, T]
        eeg_seq  = chunk.reshape(C, n_windows, window_size).transpose(1, 0, 2)

        outputs  = predict(model, eeg_seq, fs=fs, device=device, topk_proto=topk_proto)
        risk     = float(outputs["risk_prob"].item())
        raw_risks.append(risk)
        step_times.append(start)
        reports.append(build_explanation_report(outputs, topk=topk_proto))

    smooth_risks = ewma_aggregate(raw_risks, beta=ewma_beta)
    alarms       = alarm_logic(
        smooth_risks,
        threshold=alarm_threshold,
        min_consecutive=min_consecutive,
        refractory_period=refractory_period,
    )

    return {
        "raw_risks":    raw_risks,
        "smooth_risks": smooth_risks,
        "alarms":       alarms,
        "reports":      reports,
        "step_times":   step_times,
    }
