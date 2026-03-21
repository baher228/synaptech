"""Stability metrics and composite failure scoring.

All functions accept the engine-agnostic spike-train dict
``{neuron_name: [spike_time_ms, ...]}`` so they work identically with the
NumPy and Brian2 backends.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sp_stats


# ------------------------------------------------------------------ #
#  Individual metrics
# ------------------------------------------------------------------ #

def firing_rate_distribution(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
) -> dict[str, float]:
    """Per-neuron mean firing rate in Hz."""
    dur_s = duration_ms / 1000.0
    if dur_s <= 0:
        return {n: 0.0 for n in spike_trains}
    return {name: len(times) / dur_s for name, times in spike_trains.items()}


def _bin_spike_trains(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> np.ndarray:
    """Return (n_neurons, n_bins) matrix of spike counts."""
    names = sorted(spike_trains.keys())
    n_bins = max(1, int(np.ceil(duration_ms / bin_size_ms)))
    mat = np.zeros((len(names), n_bins))
    for row, name in enumerate(names):
        for t in spike_trains[name]:
            b = min(int(t / bin_size_ms), n_bins - 1)
            mat[row, b] += 1
    return mat


def network_synchrony(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Mean pairwise Pearson correlation of binned spike trains.

    Returns 0.0 when there is insufficient activity to compute correlation.
    """
    mat = _bin_spike_trains(spike_trains, duration_ms, bin_size_ms)
    active_rows = mat.sum(axis=1) > 0
    mat = mat[active_rows]
    if mat.shape[0] < 2:
        return 0.0

    corr = np.corrcoef(mat)
    np.fill_diagonal(corr, np.nan)
    mean_corr = float(np.nanmean(corr))
    return mean_corr if np.isfinite(mean_corr) else 0.0


def shannon_entropy(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Shannon entropy of the network spike-count distribution (bits)."""
    mat = _bin_spike_trains(spike_trains, duration_ms, bin_size_ms)
    total_per_bin = mat.sum(axis=0)
    total = total_per_bin.sum()
    if total == 0:
        return 0.0
    p = total_per_bin / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def pathway_fidelity(
    spike_trains: dict[str, list[float]],
    sensory_names: list[str],
    motor_names: list[str],
    duration_ms: float,
    bin_size_ms: float = 10.0,
) -> float:
    """Cross-correlation between aggregate sensory and motor spike trains.

    A high value means sensory input still drives motor output.
    Returns 0.0 if either population is silent.
    """
    n_bins = max(1, int(np.ceil(duration_ms / bin_size_ms)))
    sensory_bins = np.zeros(n_bins)
    motor_bins = np.zeros(n_bins)

    for name in sensory_names:
        for t in spike_trains.get(name, []):
            b = min(int(t / bin_size_ms), n_bins - 1)
            sensory_bins[b] += 1

    for name in motor_names:
        for t in spike_trains.get(name, []):
            b = min(int(t / bin_size_ms), n_bins - 1)
            motor_bins[b] += 1

    if sensory_bins.sum() == 0 or motor_bins.sum() == 0:
        return 0.0

    corr, _ = sp_stats.pearsonr(sensory_bins, motor_bins)
    return float(corr) if np.isfinite(corr) else 0.0


# ------------------------------------------------------------------ #
#  Baseline fingerprint
# ------------------------------------------------------------------ #

@dataclass
class BaselineFingerprint:
    mean_firing_rate: float
    std_firing_rate: float
    synchrony: float
    entropy: float
    pathway_fidelity: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_baseline(
    spike_trains: dict[str, list[float]],
    duration_ms: float,
    sensory_names: list[str],
    motor_names: list[str],
    bin_size_ms: float = 10.0,
) -> BaselineFingerprint:
    rates = firing_rate_distribution(spike_trains, duration_ms)
    rate_vals = np.array(list(rates.values()))
    return BaselineFingerprint(
        mean_firing_rate=float(np.mean(rate_vals)) if len(rate_vals) else 0.0,
        std_firing_rate=float(np.std(rate_vals)) if len(rate_vals) else 0.0,
        synchrony=network_synchrony(spike_trains, duration_ms, bin_size_ms),
        entropy=shannon_entropy(spike_trains, duration_ms, bin_size_ms),
        pathway_fidelity=pathway_fidelity(
            spike_trains, sensory_names, motor_names, duration_ms, bin_size_ms,
        ),
    )


# ------------------------------------------------------------------ #
#  Failure score
# ------------------------------------------------------------------ #

_DEFAULT_WEIGHTS = {
    "firing_rate": 0.30,
    "synchrony": 0.25,
    "entropy": 0.25,
    "pathway_fidelity": 0.20,
}


def failure_score(
    baseline: BaselineFingerprint,
    current: BaselineFingerprint,
    weights: dict[str, float] | None = None,
) -> float:
    """Composite failure score in [0, 1].

    0 = identical to baseline, 1 = total functional collapse.
    """
    w = weights or _DEFAULT_WEIGHTS

    def _safe_div(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1e-9)

    fr_div = _safe_div(current.mean_firing_rate, baseline.mean_firing_rate)
    sync_loss = max(0.0, baseline.synchrony - current.synchrony)
    ent_change = _safe_div(current.entropy, baseline.entropy)
    path_loss = max(0.0, baseline.pathway_fidelity - current.pathway_fidelity)

    raw = (
        w["firing_rate"] * min(fr_div, 1.0)
        + w["synchrony"] * min(sync_loss, 1.0)
        + w["entropy"] * min(ent_change, 1.0)
        + w["pathway_fidelity"] * min(path_loss, 1.0)
    )
    return float(np.clip(raw, 0.0, 1.0))
